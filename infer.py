"""
infer.py — German lecture slide OCR inference

Detects handwritten text/math regions and replaces them with typeset equivalents.

Usage:
    python infer.py --image slide.png --output result.png

    # With professor adaptation (n-shot):
    python infer.py --image slide.png --output result.png \
        --adapt-samples professor_samples.json --n-shot 5

    # Process a directory:
    python infer.py --image-dir slides/ --output-dir results/

    # Use a specific OCR checkpoint:
    python infer.py --image slide.png --output result.png \
        --meta-checkpoint checkpoint/maml_ocr/meta_checkpoint_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Typesetting helpers
# ---------------------------------------------------------------------------

def _get_font(size: int):
    """Return a PIL font at the given size, falling back to default."""
    from PIL import ImageFont
    candidates = [
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
        'C:/Windows/Fonts/arial.ttf',
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _fit_font_size(
    text: str,
    box_w: int,
    box_h: int,
    min_size: int = 10,
    max_size: Optional[int] = None,
    multiline: bool = False,
) -> int:
    """Find largest font size where `text` fits inside box_w x box_h.

    When `max_size` is None, derive it from `box_h` so single-line text fills
    ~85% of the box height and multi-line stacks up to ~55%. Capped at 200 to
    keep the search bounded.
    """
    from PIL import ImageDraw, Image as PILImage
    if max_size is None:
        ratio = 0.55 if multiline else 0.85
        max_size = min(200, max(min_size + 2, int(box_h * ratio)))
    if max_size <= min_size:
        return min_size

    dummy = PILImage.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy)

    def fits(size: int) -> bool:
        font = _get_font(size)
        if multiline:
            # Caller passes already-wrapped lines joined by \n. Width = widest line.
            lines = text.split('\n')
            line_h = size + 2
            if len(lines) * line_h > box_h:
                return False
            for line in lines:
                bb = draw.textbbox((0, 0), line, font=font)
                if (bb[2] - bb[0]) > box_w:
                    return False
            return True
        bb = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        return tw <= box_w and th <= box_h

    lo, hi = min_size, max_size
    best = min_size
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _erase_ink_region(
    image_pil,
    x1: int, y1: int, x2: int, y2: int,
    pad: int = 6,
    threshold: int = 195,
    dilate: int = 2,
    fill=(255, 255, 255),
) -> None:
    """Erase ONLY ink pixels (dark strokes) inside the padded bbox.

    Replaces the old rectangle-whiteout approach: nearby typeset text or
    diagram content inside the bbox stays visible because we set only
    pixels darker than `threshold` to `fill`. Dilation widens stroke
    coverage to catch anti-aliased edges.
    """
    import cv2
    from PIL import Image as PILImage
    iw, ih = image_pil.size
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(iw, x2 + pad)
    ry2 = min(ih, y2 + pad)
    if rx2 <= rx1 or ry2 <= ry1:
        return
    crop = np.array(image_pil.crop((rx1, ry1, rx2, ry2)))
    if crop.size == 0:
        return
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    if dilate > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    crop[mask > 0] = fill
    image_pil.paste(PILImage.fromarray(crop), (rx1, ry1))


def _ink_bbox(
    image_pil,
    x1: int, y1: int, x2: int, y2: int,
    threshold: int = 195,
    pad: int = 2,
) -> Optional[tuple]:
    """Return the tight bounding box of ink (dark) pixels inside the bbox.

    Detection boxes are usually much looser than the handwriting they hold.
    Shrinking to the actual ink lets the renderer anchor the typeset
    replacement where the strokes really are, instead of centering it in
    empty space. Returns (tx1, ty1, tx2, ty2) in image coordinates, or None
    when the box holds no ink.
    """
    import cv2
    iw, ih = image_pil.size
    rx1, ry1 = max(0, x1), max(0, y1)
    rx2, ry2 = min(iw, x2), min(ih, y2)
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    crop = np.array(image_pil.crop((rx1, ry1, rx2, ry2)))
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    ch, cw = crop.shape[:2]
    tx1 = rx1 + max(0, int(xs.min()) - pad)
    ty1 = ry1 + max(0, int(ys.min()) - pad)
    tx2 = rx1 + min(cw, int(xs.max()) + 1 + pad)
    ty2 = ry1 + min(ch, int(ys.max()) + 1 + pad)
    return (tx1, ty1, tx2, ty2)


def render_text_in_box(
    image_pil,
    x1: int, y1: int, x2: int, y2: int,
    text: str,
    bg_color=(255, 255, 255),
    text_color=(10, 10, 120),
    padding: int = 4,
    fill_margin: int = 6,
    erase_mode: str = 'ink',
    anchor_ink: bool = True,
):
    """Erase ink (or whiteout rect) and render typeset text inside.

    erase_mode:
      'ink'       — replace only ink pixels with bg_color (preserves nearby content)
      'rectangle' — legacy: solid bg_color rectangle covering the whole bbox
      'none'      — skip erase (caller already erased the parent region)

    anchor_ink:
      True  — anchor to the actual ink bbox (default; matches handwriting).
      False — anchor to the detector/edited box itself, so a manually
              resized region fills the box the user drew (WYSIWYG).
    """
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image_pil)

    iw, ih = image_pil.size
    # Anchor the typeset text to the actual ink, not the loose detector box,
    # so it lands on the strokes instead of floating in surrounding whitespace
    # (same idea as render_latex_in_box). Read the ink BEFORE erasing it.
    # erase_mode='none' means the parent already erased — the ink is gone, so
    # _ink_bbox returns None and layout falls back to the detector box.
    ink = _ink_bbox(image_pil, x1, y1, x2, y2) if anchor_ink else None
    if erase_mode == 'ink':
        _erase_ink_region(image_pil, x1, y1, x2, y2, pad=fill_margin, fill=bg_color)
    elif erase_mode == 'rectangle':
        fx1 = max(0, x1 - fill_margin)
        fy1 = max(0, y1 - fill_margin)
        fx2 = min(iw, x2 + fill_margin)
        fy2 = min(ih, y2 + fill_margin)
        draw.rectangle([fx1, fy1, fx2, fy2], fill=bg_color, outline=(200, 200, 200))

    lx1, ly1, lx2, ly2 = ink if ink is not None else (x1, y1, x2, y2)
    box_w = max(lx2 - lx1 - 2 * padding, 10)
    box_h = max(ly2 - ly1 - 2 * padding, 10)

    def wrap(font_size: int):
        font = _get_font(font_size)
        words = text.split()
        lines = []
        current = ''
        for word in words:
            test = (current + ' ' + word).strip()
            bb = draw.textbbox((0, 0), test, font=font)
            if bb[2] - bb[0] <= box_w:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        if not lines:
            lines = [text]
        return font, lines

    # First fit single-line, then re-fit if wrap produced multiple lines.
    font_size = _fit_font_size(text, box_w, box_h, multiline=False)
    font, lines = wrap(font_size)
    if len(lines) > 1:
        font_size = _fit_font_size('\n'.join(lines), box_w, box_h, multiline=True)
        font, lines = wrap(font_size)

    line_h = font_size + 2
    total_h = min(len(lines), max(1, box_h // line_h)) * line_h
    y0 = ly1 + padding + max(0, (box_h - total_h) // 2)

    for i, line in enumerate(lines):
        ty = y0 + i * line_h
        if ty + line_h > ly2:
            break
        draw.text((lx1 + padding, ty), line, fill=text_color, font=font)


_CJK_RE = None


def is_garbage_latex(latex: str) -> bool:
    """Return True if LaTeX output looks unrenderable / hallucinatory.

    Cheap pre-filter for obvious junk (empty, CJK, hallucination loops,
    whole-paragraph dumps). Structural LaTeX — pmatrix, multi-row ``\\``,
    aligned/cases environments — is NOT rejected here: the native
    latex+dvipng renderer (`_render_latex_native`) typesets those, and is
    itself the final arbiter — a string that fails to compile yields None
    and the caller keeps the original handwritten ink.
    """
    global _CJK_RE
    if not latex or not latex.strip():
        return True
    import re
    if _CJK_RE is None:
        _CJK_RE = re.compile(r'[　-鿿가-힯]')
    if _CJK_RE.search(latex):
        return True
    # Whole-paragraph dumps: the VLM returned prose with embedded math,
    # not a math expression. Rendering a paragraph inside a math-sized bbox
    # is unreadable — keep the original ink instead.
    text_blocks = re.findall(r'\\text\{([^}]*)\}', latex)
    if sum(len(b) for b in text_blocks) > 100:
        return True
    # pix2tex often emits these in dense nests when confused.
    nest_markers = sum(latex.count(m) for m in
                       ('\\stackrel{', '\\overrightarrow{', '\\substack{', '\\not'))
    if nest_markers >= 3:
        return True
    # Token-repetition loop = hallucination. Threshold kept loose enough that
    # genuine matrices (repeated \partial, \frac) pass.
    tokens = re.findall(r'\\[A-Za-z]+|[A-Za-z0-9]+|[^\s]', latex)
    if len(tokens) >= 6:
        from collections import Counter
        most = Counter(tokens).most_common(1)[0][1]
        if most / len(tokens) > 0.45:
            return True
    return False


def _wrap_prose_in_text(latex: str) -> str:
    """Wrap bare multi-word German prose runs inside a math expression in ``\\text{}``.

    The VLM is inconsistent: it sometimes emits prose phrases (``Dies ist der
    Fall für``, ``Waagerede Tangentialebene``) as bare tokens in math mode. In
    math mode LaTeX swallows all spaces and cannot typeset umlauts, so the
    phrase collapses into one run-together word (``DiesistderFallfr``). Wrapping
    each run of two-or-more consecutive alphabetic words in ``\\text{}`` restores
    spacing and umlauts. A *single* alphabetic token is left alone — it may be a
    multi-letter variable, a unit (``dx``) or a function name — and the
    two-word minimum keeps the rule from touching genuine math.
    """
    import re
    if not latex:
        return latex
    # Protect spans that must not be touched: existing \text{...}, row breaks
    # and backslash-commands. Protect \\ before \command so it isn't mistaken
    # for the command ``\textbackslash``-style token.
    placeholders: List[str] = []

    def _protect(m: 're.Match') -> str:
        placeholders.append(m.group(0))
        return '\x00%d\x00' % (len(placeholders) - 1)

    s = re.sub(r'\\text\{[^{}]*\}', _protect, latex)
    s = re.sub(r'\\\\', _protect, s)
    s = re.sub(r'\\[a-zA-Z]+', _protect, s)
    # A prose word: >=2 letters (incl. German umlauts), NOT followed by a
    # subscript/superscript/argument marker — that would make it math.
    word = r'[A-Za-zÄÖÜäöüß]{2,}(?![A-Za-zÄÖÜäöüß0-9_^({])'
    run = re.compile(r'(?<![_^\\])(' + word + r'(?:[ \t]+' + word + r')+)')
    s = run.sub(lambda m: '\\text{' + m.group(1) + '}', s)
    # Restore protected spans.
    s = re.sub(r'\x00(\d+)\x00', lambda m: placeholders[int(m.group(1))], s)
    return s


def _normalize_math_latex(latex: str) -> str:
    """Normalize VLM math output before garbage-filtering and rendering.

    The VLM often separates the rows of a multi-equation derivation with a
    literal newline instead of a LaTeX row break ``\\``. Left as-is, the whole
    derivation collapses onto one line. This converts inter-row newlines into
    ``\\`` so `_wrap_latex_expr` routes the block into a ``gathered``/
    ``aligned`` environment and the rows stack.

    Newlines *inside* a ``\\text{...}`` argument are prose, not row breaks —
    those are collapsed back to a single space.
    """
    import re
    if not latex:
        return latex
    expr = latex.strip()
    # The VLM sometimes emits stray '$' delimiters despite the prompt asking
    # for none. A '$' mid-expression leaves LaTeX in an unbalanced math mode
    # and the whole region fails to compile (-> left as ink). Strip them.
    expr = expr.replace('$', ' ')
    # Collapse newlines inside \text{...} prose into single spaces.
    expr = re.sub(
        r'\\text\{([^{}]*)\}',
        lambda m: '\\text{' + re.sub(r'[\r\n]+', ' ', m.group(1)) + '}',
        expr,
    )
    # If the expression already carries its own row-bearing environment,
    # don't add row breaks — that would double-break.
    if re.search(r'\\begin\{(align|gather|multline|eqnarray|matrix|pmatrix|'
                 r'bmatrix|vmatrix|cases)\*?\}', expr):
        return _wrap_prose_in_text(expr)
    # Convert remaining inter-row newlines into LaTeX row breaks. A row may
    # already end with a stray ``\\`` (VLM emitted both) — strip it first so
    # we don't produce an empty row.
    rows = [r.strip().rstrip('\\').strip() for r in re.split(r'[\r\n]+', expr)]
    rows = [r for r in rows if r]
    if len(rows) <= 1:
        return _wrap_prose_in_text(rows[0] if rows else expr)
    return _wrap_prose_in_text(' \\\\ '.join(rows))


def _wrap_latex_expr(latex: str) -> str:
    """Wrap a raw LaTeX expression into a compilable, tightly-croppable fragment.

    Inline ``$...$`` is used (not ``\\[...\\]``) so the ``standalone`` class
    crops to the math box, not a full-width display line. Multi-row content is
    routed into the right environment — all of which are valid inside ``$...$``:
      - top-level ``\\`` + ``&``  -> aligned   (alignment markers)
      - top-level ``\\``          -> gathered  (stacked centered rows)
      - a standalone display env  -> left as-is (compiles at document top level)
    Environments inside a matrix/cases keep their own ``\\`` and ``&``; the
    aligned/gathered wrapper around them is harmless.
    """
    import re
    expr = latex.strip()
    for op in ('$$', '$'):
        if expr.startswith(op) and expr.endswith(op) and len(expr) > 2 * len(op):
            expr = expr[len(op):-len(op)].strip()
            break
    if expr.startswith('\\[') and expr.endswith('\\]'):
        expr = expr[2:-2].strip()
    if re.match(r'\\begin\{(align|equation|gather|multline|eqnarray)\*?\}', expr):
        return expr
    if '\\\\' in expr:
        if '&' in expr:
            inner = '\\begin{aligned}\n%s\n\\end{aligned}' % expr
        elif '\\begin{' in expr:
            # A nested environment (e.g. a single-column pmatrix vector) owns
            # those ``\\`` — splitting on them would corrupt it. Stack the
            # whole thing centered, as before.
            inner = '\\begin{gathered}\n%s\n\\end{gathered}' % expr
        else:
            # A plain stacked derivation: left-align every row so the steps
            # read as one flush-left column (``gathered`` would centre them).
            # A leading ``&`` puts each row in ``aligned``'s left-set column;
            # ``aligned`` keeps ``\displaystyle`` so fractions stay full size.
            rows = [r.strip() for r in expr.split('\\\\') if r.strip()]
            body = ' \\\\\n'.join('& ' + r for r in rows)
            inner = '\\begin{aligned}\n%s\n\\end{aligned}' % body
    elif '&' in expr:
        inner = '\\begin{aligned}\n%s\n\\end{aligned}' % expr
    else:
        inner = expr
    return '$\\displaystyle %s$' % inner


def _render_latex_native(latex: str, dpi: int = 220, timeout: int = 20) -> Optional[np.ndarray]:
    """Compile LaTeX with the system ``latex`` + ``dvipng`` toolchain.

    Unlike matplotlib's mathtext, this renders full amsmath: pmatrix,
    multi-row ``\\``, aligned/gathered/cases environments and ``\\text{}``
    blocks. The ``preview`` package gives a tight bounding box around just
    the math. Returns a white-background RGB uint8 array at natural
    (tight-cropped) size, or None if the toolchain is missing or the
    expression fails to compile.
    """
    import os
    import shutil
    import subprocess
    import tempfile

    if shutil.which('latex') is None or shutil.which('dvipng') is None:
        return None

    doc = (
        "\\documentclass[preview,border=2pt,12pt]{standalone}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{amsmath,amssymb,amsfonts}\n"
        "\\begin{document}\n"
        + _wrap_latex_expr(latex) + "\n"
        "\\end{document}\n"
    )
    tmp = tempfile.mkdtemp(prefix='latexrender_')
    try:
        tex_path = os.path.join(tmp, 'eq.tex')
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(doc)
        subprocess.run(
            ['latex', '-interaction=nonstopmode', '-halt-on-error',
             '-output-directory', tmp, tex_path],
            capture_output=True, timeout=timeout, cwd=tmp,
        )
        dvi = os.path.join(tmp, 'eq.dvi')
        if not os.path.exists(dvi):
            return None
        png = os.path.join(tmp, 'eq.png')
        subprocess.run(
            ['dvipng', '-D', str(dpi), '-T', 'tight', '-bg', 'White',
             '-o', png, dvi],
            capture_output=True, timeout=timeout, cwd=tmp,
        )
        if not os.path.exists(png):
            return None
        from PIL import Image as PILImage
        return np.array(PILImage.open(png).convert('RGB'))
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"native LaTeX render failed: {e}")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def render_latex_in_box(
    image_pil,
    x1: int, y1: int, x2: int, y2: int,
    latex: str,
    bg_color=(255, 255, 255),
    padding: int = 4,
    erase_mode: str = 'ink',
) -> bool:
    """Render LaTeX math into a detection box.

    Uses the native latex+dvipng toolchain first (full amsmath support),
    falling back to matplotlib mathtext for simple single-line expressions.
    The render is anchored to the *actual ink* inside the detection box (not
    centered in the loose box) and sized so its height matches the
    handwriting — so the typeset math lands where the strokes were, at a
    comparable size, instead of floating tiny in whitespace.

    Returns True on success, False if the latex was garbage or neither
    renderer could typeset it. The caller falls back to leaving the original
    handwritten ink when this returns False — we never paste raw LaTeX source.
    """
    from PIL import ImageDraw, Image as PILImage

    if is_garbage_latex(latex):
        return False

    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)

    # Anchor to the tight ink box when there is ink; otherwise the loose box.
    ink = _ink_bbox(image_pil, x1, y1, x2, y2)
    atx1, aty1, atx2, aty2 = ink if ink is not None else (x1, y1, x2, y2)
    anchor_w = max(atx2 - atx1, 8)
    anchor_h = max(aty2 - aty1, 8)

    rendered = _render_latex_native(latex)
    used_native = rendered is not None
    if rendered is None:
        # matplotlib mathtext cannot do \text{} or multi-row content — it
        # garbles spacing/accents. Leave the ink rather than render junk.
        if '\\text{' in latex or '\\\\' in latex:
            return False
        rendered = _render_latex_to_array(latex, anchor_w, anchor_h)
    if rendered is None:
        return False

    nat_h, nat_w = rendered.shape[:2]
    if nat_w < 2 or nat_h < 2:
        return False

    # Re-render at a DPI that matches the ink height — avoids upscaling a
    # low-DPI bitmap (blurry). Only possible with the native toolchain.
    if used_native and nat_h >= 2:
        target_dpi = max(120, min(600, int(round(220 * anchor_h / nat_h))))
        re_rendered = _render_latex_native(latex, dpi=target_dpi)
        if re_rendered is not None and min(re_rendered.shape[:2]) >= 2:
            rendered = re_rendered
            nat_h, nat_w = rendered.shape[:2]

    # Scale to the ink height; if that overflows the ink width, fit width.
    scale = anchor_h / nat_h if nat_h else 1.0
    if nat_w * scale > anchor_w:
        scale = anchor_w / nat_w
    scale = max(0.05, min(scale, 4.0))
    new_w = max(1, int(round(nat_w * scale)))
    new_h = max(1, int(round(nat_h * scale)))
    latex_img = PILImage.fromarray(rendered).resize((new_w, new_h), PILImage.LANCZOS)

    iw, ih = image_pil.size
    if erase_mode == 'ink':
        _erase_ink_region(image_pil, x1, y1, x2, y2, pad=6, fill=bg_color)
    elif erase_mode == 'rectangle':
        fill_margin = 6
        ImageDraw.Draw(image_pil).rectangle(
            [max(0, x1 - fill_margin), max(0, y1 - fill_margin),
             min(iw, x2 + fill_margin), min(ih, y2 + fill_margin)],
            fill=bg_color, outline=(180, 180, 180),
        )
    # Anchor to the ink: left-aligned, vertically centered on the ink band.
    ox = atx1
    oy = aty1 + max(0, (anchor_h - new_h) // 2)
    image_pil.paste(latex_img, (ox, oy))
    return True


def _render_latex_to_array(latex: str, width: int, height: int) -> Optional[np.ndarray]:
    """Render LaTeX using matplotlib. Returns RGB array or None on failure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        dpi = 100
        fig_w = max(width / dpi, 0.5)
        fig_h = max(height / dpi, 0.3)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.axis('off')
        fig.patch.set_facecolor('white')

        expr = latex.strip()
        if not (expr.startswith('$') or expr.startswith('\\begin')):
            expr = f'${expr}$'

        ax.text(0.5, 0.5, expr, transform=ax.transAxes,
                ha='center', va='center', fontsize=12,
                usetex=False)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        try:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        except AttributeError:
            # matplotlib >= 3.8 removed tostring_rgb; use buffer_rgba instead
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        plt.close(fig)
        return buf
    except Exception as e:
        logger.debug(f"LaTeX render failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Line segmentation for multi-line handwritten regions
# ---------------------------------------------------------------------------

def segment_lines(region: np.ndarray, min_line_height: int = 15, gap_threshold: int = 3) -> List[tuple]:
    """
    Split a multi-line handwriting region into (y_start, y_end) row spans
    using horizontal projection profiles.

    Returns list of (y0, y1) tuples in region-local coordinates.
    """
    import cv2
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    proj = np.sum(binary > 0, axis=1).astype(int)

    # Smooth with a small window to bridge tiny gaps within a line
    kernel = np.ones(gap_threshold * 2 + 1) / (gap_threshold * 2 + 1)
    proj_smooth = np.convolve(proj, kernel, mode='same')

    lines = []
    in_line = False
    start = 0
    threshold = max(region.shape[1] * 0.005, 3)  # at least 0.5% of row width has ink

    for i, val in enumerate(proj_smooth):
        if not in_line and val >= threshold:
            in_line = True
            start = i
        elif in_line and val < threshold:
            in_line = False
            if i - start >= min_line_height:
                # Add a small vertical padding
                pad = max(4, int((i - start) * 0.1))
                lines.append((max(0, start - pad), min(region.shape[0], i + pad)))
    if in_line and region.shape[0] - start >= min_line_height:
        pad = max(4, int((region.shape[0] - start) * 0.1))
        lines.append((max(0, start - pad), region.shape[0]))

    return lines


def is_multiline(bbox: List[float], img_h: int, img_w: int) -> bool:
    """Heuristic: region is likely multi-line if its height exceeds ~5% of image height."""
    x1, y1, x2, y2 = bbox
    region_h = y2 - y1
    region_w = x2 - x1
    aspect = region_h / max(region_w, 1)
    return region_h > img_h * 0.05 and aspect > 0.15


_GERMAN_SPELL = None

# Common short German function words and abbreviations we want to count toward
# the validity score even though they're below the 3-letter threshold.
_GERMAN_SHORT = {
    'ist', 'der', 'die', 'das', 'den', 'dem', 'im', 'in', 'an', 'am', 'um',
    'zu', 'so', 'es', 'er', 'wo', 'da', 'ab', 'auf', 'aus', 'bei', 'bis',
    'für', 'mit', 'nur', 'und', 'von', 'vor', 'als', 'auch', 'noch', 'ob',
    'sei', 'sie', 'wir', 'ihr', 'uns', 'bzw',
}

# Jakob-lecture-domain compound words pyspellchecker doesn't recognize.
# Without this list, correct OCR of "Extremwertberechnungen" / "Sattelstelle"
# was being false-rejected by is_sensible_german, even though the words are
# perfectly valid German compounds. Audit on 2026-05-13 showed ~28% of dropped
# fragments were real content.
_GERMAN_JAKOB_VOCAB = {
    'extremwertberechnungen', 'extremwertberechnung',
    'extremstelle', 'extremstellen', 'extremum',
    'minimumstelle', 'minimumstellen', 'maximumstelle', 'maximumstellen',
    'sattelpunkt', 'sattelstelle', 'sattelstellen',
    'tangentialebene', 'tangentialebenen',
    'waagerechte', 'waagerecht',
    'hinreichendes', 'hinreichende', 'hinreichender',
    'kriterium', 'kriterien',
    'determinante', 'determinanten',
    'bedingung', 'bedingungen', 'nebenbedingung', 'nebenbedingungen',
    'ableitung', 'ableitungen', 'partielle', 'partiellen',
    'ordnung', 'ordnungen',
    'mehrere', 'mehreren', 'variable', 'variablen',
    'maximum', 'minimum', 'maxima', 'minima',
    'verschwinden', 'verschwindet',
    'mögliche', 'möglich', 'notwendige', 'notwendig',
    'beispiel', 'beispiele', 'folie',
    'eigenwert', 'eigenwerte', 'eigenvektor', 'eigenvektoren',
    'matrix', 'matrizen', 'vektor', 'vektoren',
    'folge', 'folgen', 'reihe', 'reihen',
    'induktion', 'integral', 'integrale', 'mehrfachintegrale',
    'gradient', 'newton', 'verfahren',
    'laplace', 'fourier', 'differentialgleichung', 'differentialgleichungen',
    'numerik',
}


def _german_word_ratio(text: str) -> float:
    """Fraction of words in `text` that are recognized German words.
    Returns -1.0 if no extractable words."""
    global _GERMAN_SPELL
    import re
    # 2+-letter alphabetic tokens (covers short function words like 'ist', 'in')
    words = [w.lower() for w in re.findall(r'[A-Za-zäöüÄÖÜß]{2,}', text)]
    if not words:
        return -1.0
    if _GERMAN_SPELL is None:
        try:
            from spellchecker import SpellChecker
            _GERMAN_SPELL = SpellChecker(language='de')
        except Exception as e:
            logger.debug(f"  pyspellchecker unavailable ({e}); skipping German check")
            _GERMAN_SPELL = False
    if _GERMAN_SPELL is False:
        return 1.0
    # Combine pyspellchecker dict with our short-word set.
    known = set(w for w in words if w in _GERMAN_SHORT)
    if _GERMAN_SPELL:
        known |= _GERMAN_SPELL.known([w for w in words if len(w) >= 3])
    return len(known) / len(words)


def is_sensible_german(text: str, min_ratio: float = 0.25) -> bool:
    """True if `text` is meaningful German content. Filters OCR gibberish
    (typeset misreads, garbled handwriting) so we don't paste nonsense over
    the original slide.

    Requires BOTH:
      (a) ≥25% of 2+-letter words recognized, AND
      (b) at least one recognized German content word with 5+ letters.

    Short function words alone (`die`, `mit`, `ist`) aren't enough — gibberish
    OCR often contains them accidentally.
    """
    global _GERMAN_SPELL
    import re
    words = [w.lower() for w in re.findall(r'[A-Za-zäöüÄÖÜß]{2,}', text)]
    if not words:
        return True  # only numbers/punctuation like 'Folie 10' or '(0,0)' — keep
    # Init dict if needed
    if _GERMAN_SPELL is None:
        try:
            from spellchecker import SpellChecker
            _GERMAN_SPELL = SpellChecker(language='de')
        except Exception:
            _GERMAN_SPELL = False
    if _GERMAN_SPELL is False:
        return True
    short_hits = sum(1 for w in words if w in _GERMAN_SHORT)
    long_words = [w for w in words if len(w) >= 5]
    spell_hits = _GERMAN_SPELL.known(long_words) if long_words else set()
    jakob_hits = {w for w in long_words if w in _GERMAN_JAKOB_VOCAB}
    long_hits = spell_hits | jakob_hits
    total_known = short_hits + len(long_hits)
    ratio = total_known / len(words)
    return (ratio >= min_ratio) and (len(long_hits) >= 1)


def is_typeset_text(region: np.ndarray) -> bool:
    """Heuristic: True if a text crop looks like printed/typeset (not handwriting).

    Typeset text on Jakob's slides has uniform stroke widths, sharp edges,
    and lives on a near-pure-white background. Handwriting has variable
    strokes, softer edges, and a slight paper-texture/gradient background.

    We score three cheap signals and call it typeset if any TWO trigger:
      1. Stroke-width CV (uniform thickness) via distance transform.
      2. Edge crispness via Laplacian variance (typeset = high sharpness).
      3. Background purity: fraction of non-ink pixels that are near-pure-white.
    """
    import cv2
    h, w = region.shape[:2]
    if h < 12 or w < 12:
        return False

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if region.ndim == 3 else region
    _, binary_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    n_fg = int(np.sum(binary_inv > 0))
    total = gray.size
    if n_fg < 80 or n_fg > 0.6 * total:
        return False

    # 1. Stroke-width coefficient of variation
    dist = cv2.distanceTransform(binary_inv, cv2.DIST_L2, 5)
    fg_dist = dist[binary_inv > 0]
    if len(fg_dist) < 50:
        return False
    mean_w = float(np.mean(fg_dist))
    std_w = float(np.std(fg_dist))
    cv_width = std_w / max(mean_w, 0.1)

    # 2. Edge crispness: Laplacian variance (higher = sharper edges)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 3. Background purity: printed content sits on near-pure white; scanned
    #    handwriting backgrounds carry slight texture/gradient.
    bg = gray[binary_inv == 0]
    bg_purity = float(np.mean(bg >= 247)) if bg.size else 0.0

    # Two-rule typeset detection (OR):
    #   (a) original strict rule — clean small typeset (titles, headers):
    #       cv<0.45 AND lap>1500.  These have cv≈0.40, lap≈2500-4000.
    #   (b) sharp-edge alone — long printed paragraphs with mixed weights:
    #       lap>2000. Handwriting on Jakob slides consistently has lap<1000,
    #       so a strong Laplacian signal alone is enough to fire.
    # Calibrated 2026-05-13 against page 9 false-negatives: typeset paragraphs
    # with cv≈0.45-0.46 (just above the strict cutoff) were leaking through
    # and getting re-rendered. Rule (b) catches those.
    #   (c) low stroke-width variation on a near-pure-white background —
    #       catches colored / low-contrast printed headers whose Laplacian
    #       is too low for rule (b).
    is_typeset = (
        (cv_width < 0.45 and lap_var > 1500.0)
        or (lap_var > 2000.0)
        or (bg_purity > 0.99 and cv_width < 0.43)
    )
    logger.debug(f"  is_typeset={is_typeset}  cv={cv_width:.2f}  lap={lap_var:.0f}  "
                 f"bg={bg_purity:.3f}  shape={h}x{w}  n_fg={n_fg}")
    return is_typeset


def is_likely_handwriting(region: np.ndarray, bbox: List[float], img_h: int, img_w: int) -> bool:
    """
    Filter out non-handwriting regions: QR codes, diagrams, photos.
    Returns False if region looks like a QR code or large image block.
    """
    x1, y1, x2, y2 = bbox
    region_h = y2 - y1
    region_w = x2 - x1
    area_frac = (region_h * region_w) / max(img_h * img_w, 1)
    aspect = region_h / max(region_w, 1)

    # Skip very square large regions — likely QR codes or images
    if area_frac > 0.08 and 0.6 < aspect < 1.7:
        # Check pixel variance: handwriting has moderate variance, QR codes are extreme
        import cv2
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        black_frac = np.mean(binary == 0)
        # QR codes have ~30-50% black pixels in a noisy high-frequency pattern
        # Handwriting has <20% black pixels on white background
        if black_frac > 0.25:
            return False

    return True


def refine_bbox_with_ink(
    img_np: np.ndarray,
    bbox: List[float],
    pad: int = 14,
    ink_threshold: int = 195,
) -> List[float]:
    """Expand a detection bbox to enclose any ink within `pad` pixels of its edges.

    Detectors often produce slightly tight (or slightly off) boxes that leave
    handwriting overflow uncovered after white-fill. This pulls a binary ink
    mask in a padded region around the box, finds its tight bbox, and unions
    it with the original — never shrinks below the input.
    """
    import cv2
    img_h, img_w = img_np.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    sx1 = max(0, x1 - pad)
    sy1 = max(0, y1 - pad)
    sx2 = min(img_w, x2 + pad)
    sy2 = min(img_h, y2 + pad)
    region = img_np[sy1:sy2, sx1:sx2]
    if region.size == 0:
        return [float(x1), float(y1), float(x2), float(y2)]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, ink_threshold, 255, cv2.THRESH_BINARY_INV)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return [float(x1), float(y1), float(x2), float(y2)]
    nx1 = sx1 + int(xs.min())
    ny1 = sy1 + int(ys.min())
    nx2 = sx1 + int(xs.max())
    ny2 = sy1 + int(ys.max())
    return [
        float(min(x1, nx1)),
        float(min(y1, ny1)),
        float(max(x2, nx2)),
        float(max(y2, ny2)),
    ]


def _dedupe_text_results(results: List[dict], overlap_threshold: float = 0.05,
                         text_similarity: float = 0.8) -> List[dict]:
    """Drop near-duplicate text overlays AFTER OCR.

    YOLO + ink-fallback sometimes produce multiple bboxes around the same
    handwritten line at different scales (e.g. one tight, one loose). The
    cross-source NMS at IoU>=0.45 catches the easy cases, but pairs at
    IoU<0.45 still pass through — each gets OCR'd to the same text and
    rendered twice on the slide.

    This second pass compares OCR output strings: if two text results
    overlap at all (>5% IoU on the smaller box) AND their lowercase texts
    are >80% similar by edit distance, keep the one with higher detection
    confidence; tie-broken by larger bbox area. Math results untouched.
    """
    text_results = [r for r in results if r.get('type') == 'text']
    other_results = [r for r in results if r.get('type') != 'text']
    if len(text_results) < 2:
        return results
    try:
        import editdistance
    except ImportError:
        return results

    def _norm(s: str) -> str:
        return ' '.join((s or '').lower().split())

    def _sim(a: str, b: str) -> float:
        a, b = _norm(a), _norm(b)
        if not a or not b:
            return 0.0
        d = editdistance.eval(a, b)
        return 1.0 - d / max(len(a), len(b))

    def _is_truncation_of(short: str, long: str) -> bool:
        """True if `short` looks like a truncated read of `long`.
        Catches the VLM-on-partial-crop case where one box was tight ("...die
        Unreiche") and a wider box around the same line read fully ("...die
        hinreichende Bedingung über die Ableitungen 2. Ordnung")."""
        if len(short) < 8 or len(long) <= len(short):
            return False
        # Long shared prefix
        prefix = 0
        for i in range(len(short)):
            if short[i] == long[i]:
                prefix += 1
            else:
                break
        if prefix >= 10 and prefix >= 0.6 * len(short):
            return True
        # Or the shorter string is a contiguous substring of the longer one
        return short in long

    def _overlap_min(b1, b2) -> float:
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        return inter / min(a1, a2)

    def _score(r: dict) -> tuple:
        bbox = r['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return (r.get('confidence', 0.0), area)

    keep = [True] * len(text_results)
    for i in range(len(text_results)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(text_results)):
            if not keep[j]:
                continue
            if _overlap_min(text_results[i]['bbox'], text_results[j]['bbox']) < overlap_threshold:
                continue
            ti = _norm(text_results[i].get('text', ''))
            tj = _norm(text_results[j].get('text', ''))
            is_dup = (
                _sim(ti, tj) >= text_similarity
                or _is_truncation_of(ti, tj)
                or _is_truncation_of(tj, ti)
            )
            if not is_dup:
                continue
            # Prefer the longer (more complete) text; tie-break on score.
            li, lj = len(ti), len(tj)
            if li > lj or (li == lj and _score(text_results[i]) >= _score(text_results[j])):
                keep[j] = False
            else:
                keep[i] = False
                break
    surviving = [r for r, k in zip(text_results, keep) if k]
    dropped = len(text_results) - len(surviving)
    if dropped:
        logger.info(f"  Post-OCR dedupe: dropped {dropped} near-duplicate text overlay(s)")

    # Drop text overlays largely covered by a typeset math region. The detector
    # sometimes emits a text box overlapping the math block; the math render
    # already owns that area, so the text overlay only collides with it.
    math_boxes = [r['bbox'] for r in other_results
                  if r.get('type') == 'math' and r.get('rendered')]
    if math_boxes:
        kept2 = []
        for r in surviving:
            b = r['bbox']
            area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
            covered = 0.0
            for mb in math_boxes:
                iw = max(0.0, min(b[2], mb[2]) - max(b[0], mb[0]))
                ih = max(0.0, min(b[3], mb[3]) - max(b[1], mb[1]))
                covered += iw * ih
            if covered / area > 0.5:
                logger.info("  Dropped text overlay covered by typeset math: "
                            f"{r.get('text', '')[:60]!r}")
            else:
                kept2.append(r)
        surviving = kept2

    return surviving + other_results


def merge_detections(dets: List[dict], iou_threshold: float = 0.45) -> List[dict]:
    """Greedily union same-class detections whose IoU >= threshold.

    YOLO and the ink fallback often produce slightly different boxes around
    the same handwriting line. Concatenating both and rendering each one
    erases the line twice and creates visible double-fills. This collapses
    those duplicates into one bbox (the union of the two) keeping the
    higher-confidence source label.
    """
    if not dets:
        return []
    items = sorted(dets, key=lambda d: -d.get('confidence', 0.0))
    used = [False] * len(items)
    merged = []
    for i, d in enumerate(items):
        if used[i]:
            continue
        used[i] = True
        x1, y1, x2, y2 = [float(v) for v in d['bbox']]
        cls = d.get('class')
        out = dict(d)
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            o = items[j]
            if o.get('class') != cls:
                continue
            ox1, oy1, ox2, oy2 = [float(v) for v in o['bbox']]
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            iw_ = max(0.0, ix2 - ix1)
            ih_ = max(0.0, iy2 - iy1)
            inter = iw_ * ih_
            a = (x2 - x1) * (y2 - y1)
            b = (ox2 - ox1) * (oy2 - oy1)
            union = a + b - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_threshold:
                x1 = min(x1, ox1)
                y1 = min(y1, oy1)
                x2 = max(x2, ox2)
                y2 = max(y2, oy2)
                used[j] = True
        out['bbox'] = [x1, y1, x2, y2]
        merged.append(out)
    return merged


def merge_row_fragments(
    dets: List[dict],
    y_overlap_frac: float = 0.5,
    x_gap_frac: float = 0.7,
    classes: tuple = ('math',),
) -> List[dict]:
    """Union same-class detections that lie on the same text row and are
    horizontally adjacent — so an equation line the detector chopped into
    several boxes becomes one box before OCR.

    Without this, the VLM transcribes each fragment in isolation and emits
    truncated LaTeX ("f_x = 3y - 3x^2 = 0 =" then nothing). Cross-source NMS
    (`merge_detections`) does NOT catch this: side-by-side fragments have
    near-zero IoU.

    Two boxes merge when, restricted to `classes` (math only by default):
      - their vertical spans overlap by >= `y_overlap_frac` of the shorter
        box (same line — stacked equations have little overlap), AND
      - the horizontal gap between them is <= `x_gap_frac` x mean box height
        (adjacent — a multi-column layout keeps columns far enough apart that
        this small gap never bridges them).
    Iterates to a fixed point so 3+ fragments chain into one box.
    """
    if not dets:
        return []
    targets = [d for d in dets if d.get('class') in classes]
    others = [d for d in dets if d.get('class') not in classes]
    changed = True
    while changed and len(targets) > 1:
        changed = False
        targets.sort(key=lambda d: float(d['bbox'][0]))
        used = [False] * len(targets)
        out = []
        for i in range(len(targets)):
            if used[i]:
                continue
            used[i] = True
            x1, y1, x2, y2 = [float(v) for v in targets[i]['bbox']]
            base = dict(targets[i])
            for j in range(i + 1, len(targets)):
                if used[j]:
                    continue
                ox1, oy1, ox2, oy2 = [float(v) for v in targets[j]['bbox']]
                v_overlap = max(0.0, min(y2, oy2) - max(y1, oy1))
                shorter = min(y2 - y1, oy2 - oy1)
                if shorter <= 0 or v_overlap / shorter < y_overlap_frac:
                    continue
                gap = max(0.0, max(x1, ox1) - min(x2, ox2))
                mean_h = 0.5 * ((y2 - y1) + (oy2 - oy1))
                if gap > x_gap_frac * mean_h:
                    continue
                x1, y1 = min(x1, ox1), min(y1, oy1)
                x2, y2 = max(x2, ox2), max(y2, oy2)
                used[j] = True
                changed = True
            base['bbox'] = [x1, y1, x2, y2]
            out.append(base)
        targets = out
    return others + targets


def detect_ink_regions(
    img_np: np.ndarray,
    existing_bboxes: List[List[float]],
    min_area_frac: float = 0.0008,
    max_area_frac: float = 0.55,
    ink_threshold: int = 200,
    margin: int = 8,
) -> List[dict]:
    """
    Fallback detector: find handwritten regions by ink density.

    Finds connected dark-pixel blobs on a light background, merges nearby
    strokes into coherent text blocks, and excludes regions already covered
    by YOLOv8 detections or identified as diagrams/photos.

    Returns list of detection dicts (same format as YOLOv8 output).
    """
    import cv2
    img_h, img_w = img_np.shape[:2]
    total_pixels = img_h * img_w

    # Threshold to binary (dark ink on white background)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, ink_threshold, 255, cv2.THRESH_BINARY_INV)

    # Merge strokes within the same text line (horizontal), then nearby lines (vertical).
    # Keep horizontal kernel modest so wide headers don't fuse into slide-spanning blobs.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 4))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 18))
    merged = cv2.dilate(binary, h_kernel, iterations=2)
    merged = cv2.dilate(merged, v_kernel, iterations=2)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    new_dets = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area_frac = (bw * bh) / total_pixels

        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue

        # Skip horizontal rules / thin decorative lines
        aspect = bh / max(bw, 1)
        if aspect < 0.02:
            continue

        # Skip slide-spanning blobs (full-width header/footer merge artifacts)
        if bw > img_w * 0.85:
            continue

        # Add margin
        x1 = max(0, bx - margin)
        y1 = max(0, by - margin)
        x2 = min(img_w, bx + bw + margin)
        y2 = min(img_h, by + bh + margin)

        # Skip if mostly covered by an existing YOLOv8 detection
        skip = False
        for eb in existing_bboxes:
            ex1, ey1, ex2, ey2 = [int(v) for v in eb]
            inter_w = max(0, min(x2, ex2) - max(x1, ex1))
            inter_h = max(0, min(y2, ey2) - max(y1, ey1))
            inter_area = inter_w * inter_h
            region_area = (x2 - x1) * (y2 - y1)
            if region_area > 0 and inter_area / region_area > 0.6:
                skip = True
                break
        if skip:
            continue

        # Require minimum ink density — filters out blank / near-blank regions
        region_crop = binary[y1:y2, x1:x2]
        ink_density = np.mean(region_crop > 0)
        if ink_density < 0.005:
            continue

        # Filter out diagrams / photos using two checks:
        # 1. Grayscale diversity: handwriting on white paper is mostly pure white (>200)
        #    or pure black (<60). Diagrams/photos have many intermediate gray values.
        region_gray = gray[y1:y2, x1:x2]
        mid_gray_frac = np.mean((region_gray > 60) & (region_gray < 200))
        if mid_gray_frac > 0.10:
            logger.debug(f"  Ink detector: skipping likely diagram/photo at "
                         f"({x1},{y1},{x2},{y2}) mid_gray={mid_gray_frac:.2f}")
            continue

        # 2. Edge complexity: diagrams have many connected curved edges;
        #    handwriting text lines are mostly horizontal with sparse edges.
        edges = cv2.Canny(region_gray, 50, 150)
        edge_density = np.mean(edges > 0)
        raw_ink = np.mean(region_crop > 0)
        if edge_density > 0.12 and raw_ink > 0.18:
            logger.debug(f"  Ink detector: skipping likely diagram at ({x1},{y1},{x2},{y2}) "
                         f"edge={edge_density:.2f} ink={raw_ink:.2f}")
            continue

        new_dets.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': 0.5,
            'class': 'text',
            'source': 'ink_detector',
        })

    logger.debug(f"  Ink detector found {len(new_dets)} additional regions")
    return new_dets


_LATEX_TO_UNICODE = {
    '\\Delta': 'Δ', '\\alpha': 'α', '\\beta': 'β', '\\gamma': 'γ', '\\delta': 'δ',
    '\\epsilon': 'ε', '\\zeta': 'ζ', '\\eta': 'η', '\\theta': 'θ', '\\iota': 'ι',
    '\\kappa': 'κ', '\\lambda': 'λ', '\\mu': 'μ', '\\nu': 'ν', '\\xi': 'ξ',
    '\\pi': 'π', '\\rho': 'ρ', '\\sigma': 'σ', '\\tau': 'τ', '\\phi': 'φ',
    '\\chi': 'χ', '\\psi': 'ψ', '\\omega': 'ω',
    '\\cdot': '·', '\\times': '×', '\\div': '÷', '\\pm': '±', '\\mp': '∓',
    '\\leq': '≤', '\\geq': '≥', '\\neq': '≠', '\\approx': '≈', '\\equiv': '≡',
    '\\rightarrow': '→', '\\leftarrow': '←', '\\Rightarrow': '⇒', '\\Leftarrow': '⇐',
    '\\infty': '∞', '\\partial': '∂', '\\nabla': '∇',
    '\\sum': '∑', '\\prod': '∏', '\\int': '∫', '\\sqrt': '√',
    '\\in': '∈', '\\notin': '∉', '\\subset': '⊂', '\\supset': '⊃',
    '\\forall': '∀', '\\exists': '∃',
}


def _clean_vlm_text(text: str) -> str:
    """Tidy VLM text output for plain-text rendering.

    The VLM is prompted to omit `$` delimiters and to use LaTeX only for
    math content, but it sometimes leaks `$\\Delta$` or bare `\\alpha` into
    otherwise-prose lines. We unwrap `$...$` and substitute common LaTeX
    Greek/operator tokens with Unicode equivalents so the rendered overlay
    reads naturally instead of showing `$\\Delta$ ist Determinante`.
    """
    if not text:
        return text
    import re
    # Drop $ delimiters
    text = re.sub(r'\$([^\$]+)\$', r'\1', text)
    text = text.replace('$', '')
    # Substitute LaTeX tokens with Unicode (longest first to avoid partial overlap)
    for tok in sorted(_LATEX_TO_UNICODE.keys(), key=len, reverse=True):
        if tok in text:
            text = text.replace(tok, _LATEX_TO_UNICODE[tok])
    return text


def _has_raw_latex_artifacts(text: str) -> bool:
    """True if text has LaTeX markup that survived `_clean_vlm_text` and would
    render as ugly raw code in a plain-text overlay.

    Catches:
      - bare backslash commands (e.g. `\\frac`, `\\sqrt`) — _clean_vlm_text only
        converts a curated set of Greek/operator tokens.
      - curly-brace subscripts / superscripts: `_{...}`, `^{...}`
      - chained underscores like `H_f`, `f_{xx}`, `f_xy`, `f_yx` that signal
        handwritten math wrongly classified as text by the ink-fallback
        detector (e.g. the "Hessche Matrix: H_f(x;y) = (f_xx f_yx ..." case
        from page 9). Two or more such tokens in one fragment is the trigger.

    Such results are dropped (left as ink) rather than rendered, since neither
    plain-text rendering nor matplotlib mathtext produces a clean output.
    """
    if not text:
        return False
    import re
    if re.search(r'\\[A-Za-z]+', text):
        return True
    if re.search(r'[_^]\{', text):
        return True
    if len(re.findall(r'[A-Za-z]_[A-Za-z]{1,4}', text)) >= 2:
        return True
    return False


def _is_math_heavy(text: str) -> bool:
    """True if the cleaned text looks like a math expression more than prose.

    Used to re-route text-class crops whose VLM output came back as raw
    LaTeX (e.g. handwritten math wrongly labeled 'text' by the ink-fallback
    detector) through the matplotlib math renderer instead of plain text.
    """
    if not text:
        return False
    import re
    stripped = text.strip()
    if not stripped:
        return False
    letters = sum(c.isalpha() for c in stripped)
    digits = sum(c.isdigit() for c in stripped)
    math_chars = sum(1 for c in stripped if c in '+-=<>^_{}[]/\\|·×÷≤≥≠≈∞∂∇∑∏∫√')
    # German words have at least 3-letter alphabetic runs separated by spaces.
    long_word_count = len([w for w in re.findall(r'[A-Za-zäöüÄÖÜß]+', stripped) if len(w) >= 4])
    total = max(1, len(stripped))
    # High math-char density AND few prose-like words → treat as math.
    return (math_chars / total) > 0.20 and long_word_count <= 1


def _looks_like_math(text: str, region: np.ndarray) -> bool:
    """
    Heuristic: does this OCR result (or region image) look like a math expression?
    Uses text character ratios and basic image structure.
    """
    if not text:
        return False
    alpha = sum(c.isalpha() for c in text)
    total = len(text)
    math_chars = sum(1 for c in text if c in '+-=/<>^_{}[]()\\|∫∑∏√·×÷')
    # Looks like math if more than 25% math chars, or very short and has digits
    if total > 0 and math_chars / total > 0.25:
        return True
    if total < 20 and sum(c.isdigit() for c in text) / max(total, 1) > 0.3:
        return True
    return False


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def load_pipeline(
    detector_path: str,
    meta_checkpoint: Optional[str],
    ocr_fallback: str,
    device: str,
    adapt_lr: float = 0.01,
    detector_imgsz: int = 960,
    ocr_backend: str = 'trocr',
    vlm_adapter: Optional[str] = None,
    vlm_mlx_model: Optional[str] = None,
):
    """Load detector + OCR + math OCR models.

    `ocr_backend='trocr'`: MAMLOCRWrapper for text, TAMER/pix2tex for math.
    `ocr_backend='vlm'`:   Qwen3-VL-8B for both. Math uses the same VLM
                           instance with a LaTeX prompt (no TAMER/pix2tex).
                           On CUDA this is the bitsandbytes 8-bit
                           `VLMOCRBackend`; on Apple Silicon (mps/cpu) it is the
                           `MLXVLMOCRBackend` instead — `bitsandbytes` is
                           CUDA-only. `vlm_adapter` layers a Phase 2 LoRA on the
                           CUDA backend; the MLX backend bakes the adapter into
                           the model file instead (see `vlm_mlx_model`).
    """
    from baseline.baseline_pipeline import YOLOv8Detector

    detector = YOLOv8Detector(weights=detector_path, device=device, imgsz=detector_imgsz)

    if ocr_backend == 'vlm':
        if device == 'cuda':
            from models.vlm_ocr import VLMOCRBackend
            ocr = VLMOCRBackend(device=device, adapter_path=vlm_adapter)
            logger.info("OCR backend: Qwen3-VL-8B (CUDA, 8-bit, shared for text and math)")
        else:
            from models.vlm_ocr_mlx import MLXVLMOCRBackend
            if vlm_adapter:
                logger.warning(
                    "--vlm-adapter is ignored on the MLX backend; bake the adapter "
                    "into the model with scripts/fuse_and_convert_vlm_mlx.py and pass "
                    "the result via --vlm-mlx-model.")
            ocr = MLXVLMOCRBackend(
                model_id=(vlm_mlx_model
                          or 'lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit'),
                device=device,
            )
            logger.info(f"OCR backend: MLX Qwen3-VL '{ocr.model_id}' on {device} "
                        "(shared for text and math)")
        # Math is handled by the same VLM instance with mode='math' — return it
        # for both slots so callers can detect the shared backend via identity.
        math_ocr = ocr
    else:
        from models.math_ocr_tamer import TAMERMathOCR
        ocr = _load_ocr(meta_checkpoint, ocr_fallback, device, inner_lr=adapt_lr)
        math_ocr = TAMERMathOCR(device=device)
        logger.info("OCR backend: TrOCR (text) + TAMER/pix2tex (math)")

    return detector, ocr, math_ocr


def _load_ocr(meta_checkpoint: Optional[str], fallback_path: str, device: str, inner_lr: float = 0.01):
    """Load MAMLOCRWrapper with checkpoint, or plain TrOCR as fallback."""
    from models.meta_learning_ocr import MAMLOCRWrapper
    import torch

    wrapper = MAMLOCRWrapper(base_model_path=fallback_path, device=device, inner_lr=inner_lr)

    if meta_checkpoint and Path(meta_checkpoint).exists():
        ckpt = torch.load(meta_checkpoint, map_location=device, weights_only=False)
        wrapper.meta_model.load_state_dict(ckpt['meta_model_state'])
        epoch = ckpt.get('epoch', '?')
        val_cer = ckpt.get('val_cer', float('nan'))
        logger.info(f"Loaded meta-checkpoint: epoch={epoch+1 if isinstance(epoch,int) else epoch}, "
                    f"val_CER={val_cer*100:.2f}%")
    else:
        logger.info(f"No meta-checkpoint found; using base OCR model from {fallback_path}")

    return wrapper


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_infer(
    image_path: Path,
    output_path: Path,
    detector,
    ocr,
    math_ocr,
    adapt_samples: Optional[List[dict]] = None,
    n_shot: int = 5,
    conf_threshold: float = 0.35,
    annotate_only: bool = False,
    force_tamer: bool = False,
    postprocess_german: bool = True,
    corrector=None,
    erase_mode: str = 'ink',
    enable_math_ocr: bool = False,
) -> List[dict]:
    """
    Run full OCR pipeline on one slide image and write typeset output.

    Args:
        image_path:     Input slide image.
        output_path:    Where to save the result.
        detector:       YOLOv8Detector instance.
        ocr:            MAMLOCRWrapper instance.
        math_ocr:       TAMERMathOCR instance.
        adapt_samples:  Optional list of {image, text} dicts for professor adaptation.
        n_shot:         Adaptation steps if adapt_samples is provided.
        conf_threshold: Detection confidence cutoff.
        annotate_only:  If True, draw boxes instead of replacing content.

    Returns:
        List of detection result dicts.
    """
    from PIL import Image as PILImage
    from utils.image_utils import load_image, extract_region

    logger.info(f"Processing: {image_path}")
    t0 = time.time()

    img_np = load_image(image_path, mode='rgb')
    img_pil = PILImage.fromarray(img_np)

    # Step 1: Detect
    detector.conf_threshold = conf_threshold
    detections = detector.detect(img_np)
    img_h_pre, img_w_pre = img_np.shape[:2]
    slide_area = float(img_h_pre * img_w_pre)
    pruned = []
    for d in detections:
        x1f, y1f, x2f, y2f = d['bbox']
        bw = x2f - x1f
        bh = y2f - y1f
        area_frac = (bw * bh) / slide_area
        aspect = bh / max(1.0, bw)
        # Prune square-ish big blobs (likely diagrams/photos) OR things that
        # are both very wide AND very tall. Wide-thin equation bands survive.
        is_big_squarish = area_frac > 0.5 and aspect > 0.4
        is_wide_and_tall = bw > 0.95 * img_w_pre and bh > 0.4 * img_h_pre
        if is_big_squarish or is_wide_and_tall:
            logger.debug(f"  Suppressing oversized detection ({d['class']}): "
                         f"{int(bw)}x{int(bh)} = {area_frac*100:.1f}% of slide, "
                         f"aspect={aspect:.2f}")
            continue
        pruned.append(d)
    if len(pruned) != len(detections):
        logger.info(f"  Pruned {len(detections)-len(pruned)} oversized detections")
    detections = pruned

    # Refine each YOLO bbox to enclose nearby ink — handles tight/loose detections.
    for d in detections:
        d['bbox'] = refine_bbox_with_ink(img_np, d['bbox'])

    logger.info(f"  Detected {len(detections)} regions "
                f"({sum(1 for d in detections if d['class']=='text')} text, "
                f"{sum(1 for d in detections if d['class']=='math')} math)")

    # Fallback: ink-based detection for handwriting YOLOv8 misses
    existing_bboxes = [d['bbox'] for d in detections]
    ink_dets = detect_ink_regions(img_np, existing_bboxes)
    if ink_dets:
        logger.info(f"  Ink detector added {len(ink_dets)} region(s) missed by YOLOv8")
        detections = detections + ink_dets

    # Cross-source NMS — collapse YOLO+ink duplicates that survived the
    # 60%-coverage check inside detect_ink_regions but still overlap heavily.
    n_pre_merge = len(detections)
    detections = merge_detections(detections, iou_threshold=0.45)
    if len(detections) != n_pre_merge:
        logger.info(f"  Merged {n_pre_merge - len(detections)} overlapping detection(s) "
                    f"(cross-source NMS)")

    # Stitch math equation lines the detector fragmented into adjacent boxes,
    # so the VLM sees a whole line and doesn't emit truncated LaTeX.
    n_pre_frag = len(detections)
    detections = merge_row_fragments(detections)
    if len(detections) != n_pre_frag:
        logger.info(f"  Stitched {n_pre_frag - len(detections)} fragmented math "
                    f"detection(s) into whole equation lines")

    # Filter out typeset (printed) text regions — we only want to replace
    # actual handwriting. Typeset detections previously caused OCR to emit
    # gibberish (e.g. 'Hinreichendes Kriterium:' → 'Itinveicheudes Koiterium:')
    # which then got pasted over the original clean text.
    # A wide region flush against the top (or bottom) page edge is the printed
    # slide header/footer band — drop it outright so it is never OCR'd and
    # re-typeset. Body handwriting starts well below the top edge, so keying on
    # a tiny top y1 keeps legitimate near-top handwriting safe.
    ph, pw = img_np.shape[:2]

    def _is_printed_header(d) -> bool:
        bx1, by1, bx2, by2 = d['bbox']
        wide = (bx2 - bx1) >= 0.20 * pw
        at_top = by1 <= 0.045 * ph
        at_bottom = by2 >= 0.96 * ph
        return wide and (at_top or at_bottom)

    n_before = len(detections)
    detections = [
        d for d in detections
        if not _is_printed_header(d)
        and not is_typeset_text(img_np[int(d['bbox'][1]):int(d['bbox'][3]),
                                       int(d['bbox'][0]):int(d['bbox'][2])])
    ]
    n_filtered = n_before - len(detections)
    if n_filtered:
        logger.info(f"  Typeset filter dropped {n_filtered} region(s) "
                    f"(printed text, leaving original visible)")

    if not detections:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img_pil.save(str(output_path))
        logger.info(f"  No regions found. Saved copy -> {output_path}")
        return []

    # Note: adaptation is performed once in main() before the loop, not per-slide.

    text_dets = [d for d in detections if d['class'] == 'text']
    math_dets = [d for d in detections if d['class'] == 'math']

    img_h, img_w = img_np.shape[:2]
    results = []
    use_adapted = adapt_samples is not None

    # Detect VLM backend by feature, not isinstance, so this file doesn't
    # need to import VLMOCRBackend just to type-check.
    is_vlm_backend = hasattr(ocr, '_run_one') and hasattr(ocr, 'model_id')

    # Step 3: Text OCR — split multi-line regions into individual lines, run OCR,
    # but DO NOT render yet. Rendering happens in _render_text_results below, after
    # an optional LLM correction pass.
    for det in text_dets:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        region = img_np[y1:y2, x1:x2]

        if not is_likely_handwriting(region, det['bbox'], img_h, img_w):
            logger.debug(f"  Skipping non-handwriting region at ({x1},{y1},{x2},{y2})")
            continue

        if is_vlm_backend:
            # VLM handles multi-line input natively — skip the projection-based
            # line segmentation, pass the whole region as one prompt.
            line_spans = [(0, region.shape[0])]
            line_crops = [region]
            line_texts = ocr.predict(line_crops, mode='text')
            line_texts = [_clean_vlm_text(t) for t in line_texts]
        else:
            if is_multiline(det['bbox'], img_h, img_w):
                line_spans = segment_lines(region)
                logger.debug(f"  Multi-line region split into {len(line_spans)} lines")
            else:
                line_spans = [(0, region.shape[0])]

            line_crops = [region[ly0:ly1, :] for ly0, ly1 in line_spans if ly1 > ly0]
            if line_crops:
                line_texts = ocr.predict(line_crops, use_adapted=use_adapted,
                                         postprocess_german=postprocess_german)
            else:
                line_texts = []

        full_text = ' '.join(line_texts)

        # is_sensible_german is calibrated against TrOCR character-noise; VLM
        # output is already clean German, so we skip the gate when using VLM.
        if not is_vlm_backend and full_text.strip() and not is_sensible_german(full_text):
            logger.debug(f"  Dropping non-German OCR result: {full_text!r}")
            continue

        # When using VLM + math OCR, a text-class crop whose VLM output is
        # heavily math (e.g. handwritten math wrongly labeled 'text' by the
        # ink-fallback detector) should be routed through the matplotlib
        # LaTeX renderer instead of being rendered as raw "f_{xx}" text.
        if is_vlm_backend and enable_math_ocr and _is_math_heavy(full_text):
            math_latex = _normalize_math_latex(full_text)
            results.append({
                **det, 'type': 'math',
                'latex': math_latex, 'text': math_latex, 'rendered': True,
            })
            continue

        # Mixed German+math fragments (e.g. "Hessche Matrix: H_f(x;y) = (f_xx ...")
        # render badly both as plain text (raw LaTeX showing) and as math
        # (matplotlib mathtext can't handle pmatrix or text mixed in). Leave
        # the original handwriting visible instead.
        if is_vlm_backend and _has_raw_latex_artifacts(full_text):
            logger.debug(f"  Dropping mixed text/LaTeX VLM output (leaving ink): {full_text!r}")
            continue

        results.append({
            **det,
            'type': 'text',
            'text': full_text,
            'text_orig': full_text,
            'line_spans': line_spans,
            'line_texts': list(line_texts),
            'line_texts_orig': list(line_texts),
        })

    # Step 4: Math regions.
    # Default (enable_math_ocr=False): leave original handwritten ink visible.
    # With --enable-math-ocr: run pix2tex/TAMER, accept only non-garbage LaTeX,
    # render typeset math via matplotlib. Bad LaTeX falls back to ink-only.
    if enable_math_ocr and math_dets:
        n_ok = 0
        for det in math_dets:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            crop = img_np[y1:y2, x1:x2]
            if crop.size == 0:
                results.append({**det, 'type': 'math', 'latex': '',
                                'text': '', 'rendered': False})
                continue
            try:
                if is_vlm_backend:
                    # math_ocr is the same VLMOCRBackend instance as ocr here.
                    latex = (math_ocr.predict([crop], mode='math') or [''])[0]
                else:
                    latex = math_ocr.recognize(crop) or ''
            except Exception as e:
                logger.debug(f"  Math OCR failed at ({x1},{y1},{x2},{y2}): {e}")
                latex = ''
            # Turn literal-newline-separated derivations into stacked LaTeX
            # rows before the garbage filter and renderer see them.
            latex = _normalize_math_latex(latex)
            if latex and not is_garbage_latex(latex):
                results.append({**det, 'type': 'math', 'latex': latex,
                                'text': latex, 'rendered': True})
                n_ok += 1
            else:
                if latex:
                    logger.debug(f"  Math OCR rejected garbage: {latex!r}")
                results.append({**det, 'type': 'math', 'latex': '',
                                'text': '', 'rendered': False})
        logger.info(f"  Math OCR: typeset {n_ok}/{len(math_dets)} math region(s) "
                    f"(rest left as ink)")
    else:
        for det in math_dets:
            results.append({**det, 'type': 'math', 'text': '', 'rendered': False})

    # Step 5: Optional LLM correction on text-only fragments, then render.
    # Skipped for VLM backend — its output is already clean German and the
    # text-only LLM corrector can only degrade it (no vision context to
    # disambiguate good vs. bad characters).
    if corrector is not None and not annotate_only and not is_vlm_backend:
        try:
            _apply_llm_correction(corrector, results)
        except Exception as e:
            logger.warning(f"  LLM correction failed; using raw OCR. ({e})")

    # Drop near-duplicate text overlays before render (catches what cross-source
    # NMS at IoU>=0.45 missed — two boxes around the same line that OCR to the
    # same string).
    results = _dedupe_text_results(results)

    if not annotate_only:
        _render_text_results(img_pil, results, erase_mode=erase_mode)
    else:
        _draw_annotations(img_pil, results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_pil.save(str(output_path))
    elapsed = time.time() - t0
    logger.info(f"  Saved -> {output_path} ({elapsed:.1f}s)")

    return results


def _apply_llm_correction(corrector, results: List[dict]) -> None:
    """Send each text line as a fragment to the LLM; mutate `results` in-place
    with corrected line_texts. Math results are skipped entirely."""
    fragments = []
    locs = []  # (result_idx, line_idx)
    for ri, r in enumerate(results):
        if r.get('type') != 'text':
            continue
        for li, t in enumerate(r.get('line_texts', [])):
            if not t.strip():
                continue
            fragments.append({'idx': len(fragments), 'text': t})
            locs.append((ri, li))
    if not fragments:
        return
    logger.info(f"  LLM correcting {len(fragments)} text fragments...")
    corrected = corrector.correct_slide(fragments)
    n_changed = 0
    for frag, (ri, li) in zip(corrected, locs):
        new_t = frag.get('text', '')
        if new_t != results[ri]['line_texts'][li]:
            results[ri]['line_texts'][li] = new_t
            n_changed += 1
    # Rebuild full 'text' field
    for r in results:
        if r.get('type') == 'text' and 'line_texts' in r:
            r['text'] = ' '.join(r['line_texts'])
    logger.info(f"  LLM applied {n_changed} corrections (of {len(fragments)} fragments)")


def _render_text_results(img_pil, results: List[dict], erase_mode: str = 'ink') -> None:
    """Idempotent: erase each region's ink and render typeset text/math.
    Math results without a valid 'latex' field are left as original ink.

    A text region with ``r['fit'] == 'box'`` (set by the slide editor once a
    region has been manually moved/resized) renders anchored to its own bbox
    instead of the original handwriting ink — WYSIWYG for edited regions,
    while freshly-inferred regions keep the ink-anchored layout. Math regions
    always stay ink-anchored: the LaTeX renderer's DPI/scale fit assumes the
    box is close to the ink size, and a much larger user-drawn box scales the
    typeset expression far past its neighbours."""
    from PIL import ImageDraw
    iw, ih = img_pil.size
    for r in results:
        rtype = r.get('type')
        if rtype == 'math':
            if r.get('rendered') and r.get('latex'):
                x1, y1, x2, y2 = [int(v) for v in r['bbox']]
                ok = render_latex_in_box(
                    img_pil, x1, y1, x2, y2, r['latex'], erase_mode=erase_mode,
                )
                if not ok:
                    # matplotlib parse failed despite is_garbage_latex passing — leave ink.
                    r['rendered'] = False
            continue
        if rtype != 'text':
            continue
        anchor_ink = r.get('fit') != 'box'
        x1, y1, x2, y2 = [int(v) for v in r['bbox']]
        line_spans = r.get('line_spans') or [(0, y2 - y1)]
        line_texts = r.get('line_texts') or []
        full_text = r.get('text', '').strip()
        if not full_text and not any(t.strip() for t in line_texts):
            continue

        if anchor_ink and len(line_spans) > 1 and line_texts:
            fm = 6
            if erase_mode == 'ink':
                _erase_ink_region(img_pil, x1, y1, x2, y2, pad=fm)
            else:
                ImageDraw.Draw(img_pil).rectangle(
                    [max(0, x1 - fm), max(0, y1 - fm),
                     min(iw, x2 + fm), min(ih, y2 + fm)],
                    fill=(255, 255, 255),
                )
            for (ly0, ly1), text in zip(line_spans, line_texts):
                if not text.strip():
                    continue
                # Parent already erased; per-line render only draws the text.
                render_text_in_box(img_pil, x1, y1 + ly0, x2, y1 + ly1, text,
                                   bg_color=(255, 255, 255), fill_margin=0,
                                   erase_mode='none')
        else:
            # Box-anchored regions render as one block: `text` may still
            # contain '\n' from line_texts, but render_text_in_box's wrap()
            # splits on whitespace and re-wraps to the (new) box width.
            render_text_in_box(img_pil, x1, y1, x2, y2, full_text,
                               erase_mode=erase_mode, anchor_ink=anchor_ink)


def _draw_annotations(img_pil, results: List[dict]):
    """Draw colored boxes + labels for annotate-only mode."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img_pil)
    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r['bbox']]
        color = (0, 180, 0) if r['class'] == 'text' else (180, 0, 180)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{r['class']}: {r.get('text','')[:25]}"
        font = _get_font(12)
        draw.text((x1 + 2, max(0, y1 - 14)), label, fill=color, font=font)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Lecture slide OCR: replace handwriting with typeset text'
    )

    in_group = parser.add_mutually_exclusive_group(required=True)
    in_group.add_argument('--image', type=Path, help='Single input image')
    in_group.add_argument('--image-dir', type=Path, help='Directory of images to process')

    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument('--output', type=Path, default=None,
                           help='Output path for single image (default: <input>_out.<ext>)')
    out_group.add_argument('--output-dir', type=Path, default=None,
                           help='Output directory for batch mode')

    parser.add_argument('--detector-path', type=str,
                        default='runs/detect/runs/jakob_detector_v3/hires1536-3/weights/best.pt',
                        help='YOLOv8 checkpoint path')
    parser.add_argument('--detector-imgsz', type=int, default=960,
                        help='Inference resolution for YOLOv8 (must match training: 960 for v2_hires_w0, 640 for v1/v2)')
    parser.add_argument('--meta-checkpoint', type=str, default=None,
                        help='Reptile/MAML meta-learned OCR checkpoint (overlays on --ocr-fallback). '
                             'Default None = use --ocr-fallback weights directly.')
    parser.add_argument('--ocr-fallback', type=str,
                        default='checkpoint/trocr_jakob/best',
                        help='HF-format TrOCR checkpoint loaded as the OCR base. '
                             'Default: Jakob fine-tune (val CER 8.82%%). '
                             'Use checkpoint/trocr_german/best for IAM-only baseline.')
    parser.add_argument('--adapt-samples', type=Path, default=None,
                        help='JSON file with professor samples [{image, text}, ...] for adaptation')
    parser.add_argument('--n-shot', type=int, default=5,
                        help='Number of adaptation samples to use')
    parser.add_argument('--adapt-steps', type=int, default=2,
                        help='Number of inner-loop adaptation steps (lower=safer, 5 diverges)')
    parser.add_argument('--adapt-lr', type=float, default=0.01,
                        help='Inner-loop learning rate for adaptation (lower=safer for more steps)')
    parser.add_argument('--conf', type=float, default=0.08,
                        help='Detection confidence threshold (low default = higher recall)')
    parser.add_argument('--annotate-only', action='store_true',
                        help='Draw detection boxes instead of replacing content')
    parser.add_argument('--force-tamer', action='store_true',
                        help='Route ALL regions through TAMER math OCR (for math-heavy slides)')
    parser.add_argument('--no-german-postproc', action='store_true',
                        help='Disable German-specific OCR postprocessing (use for English slides)')
    parser.add_argument('--device', type=str, default='auto',
                        help="Compute device: 'auto' (cuda>mps>cpu), 'cuda', 'mps', or 'cpu'.")
    parser.add_argument('--vlm-mlx-model', type=str,
                        default='lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit',
                        help='MLX VLM model id or local path (used only on mps/cpu, '
                             'i.e. Apple Silicon). Ignored on CUDA.')
    parser.add_argument('--save-json', action='store_true',
                        help='Also save OCR results as JSON alongside output image')
    parser.add_argument('--llm-correct', action=argparse.BooleanOptionalAction, default=True,
                        help='Use local LLM (Qwen3-8B by default) to correct German OCR text. '
                             'Math regions are never sent to the LLM. Disable with --no-llm-correct.')
    parser.add_argument('--llm-model', type=str, default='Qwen/Qwen3-8B',
                        help='HF model id for the local LLM corrector.')
    parser.add_argument('--llm-8bit', action=argparse.BooleanOptionalAction, default=True,
                        help='Load LLM in 8-bit via bitsandbytes (default ON to coexist with '
                             'OCR models on 16GB GPUs). Use --no-llm-8bit on bigger GPUs.')
    parser.add_argument('--legacy-render', action='store_true',
                        help='Use legacy solid-rectangle whiteout (covers everything inside '
                             'each bbox). Default is ink-aware erase, which only replaces '
                             'dark stroke pixels and preserves nearby typeset content.')
    parser.add_argument('--enable-math-ocr', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Run math regions through math OCR and render typeset LaTeX. '
                             'Default depends on backend: ON for --ocr-backend vlm (Qwen3-VL '
                             'reliably produces LaTeX), OFF for --ocr-backend trocr (pix2tex/TAMER '
                             'often hallucinate). Garbage LaTeX is filtered and falls back to ink.')
    parser.add_argument('--ocr-backend', type=str, default='vlm',
                        choices=['trocr', 'vlm'],
                        help='Which OCR backend to use. "vlm" (default): Qwen3-VL-8B on each '
                             'crop (skips line segmentation, is_sensible_german, LLM corrector); '
                             '~2% CER zero-shot on Jakob handwriting (~4x better than TrOCR). '
                             'With --enable-math-ocr the VLM also handles math crops (replaces '
                             'pix2tex). "trocr": TrOCR + Reptile meta-learning + Jakob fine-tune '
                             '+ LLM corrector — the staged-pipeline baseline; without the Jakob '
                             'fine-tune it falls back to base trocr-large-handwritten (English) '
                             'and badly garbles German cursive.')
    parser.add_argument('--vlm-adapter', type=str, default=None,
                        help='Path to a Phase 2 per-professor LoRA adapter (e.g. '
                             'checkpoint/vlm_jakob_lora/best). Only used with '
                             '--ocr-backend vlm; layers the adapter on the VLM.')

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve 'auto'/'cuda'/'mps' to a concrete device. On the CUDA server this
    # returns 'cuda' unchanged; on an Apple Silicon Mac it returns 'mps'.
    from utils.device import get_device
    args.device = get_device(args.device)
    logger.info(f"Compute device: {args.device}")

    # Load models once
    logger.info("Loading pipeline models...")
    detector, ocr, math_ocr = load_pipeline(
        detector_path=args.detector_path,
        meta_checkpoint=args.meta_checkpoint,
        ocr_fallback=args.ocr_fallback,
        device=args.device,
        adapt_lr=args.adapt_lr,
        detector_imgsz=args.detector_imgsz,
        ocr_backend=args.ocr_backend,
        vlm_adapter=args.vlm_adapter,
        vlm_mlx_model=args.vlm_mlx_model,
    )

    using_vlm = args.ocr_backend == 'vlm'
    if args.enable_math_ocr is None:
        # VLM is reliable at LaTeX; TrOCR path's pix2tex/TAMER is not.
        args.enable_math_ocr = using_vlm

    # Load optional adaptation samples and adapt ONCE before the loop.
    # Adaptation only applies to the TrOCR/Reptile path; VLM ignores it in Phase 1.
    adapt_samples = None
    if args.adapt_samples and args.adapt_samples.exists():
        if using_vlm:
            logger.info("Skipping Reptile adaptation: VLM backend has no inner-loop adapter "
                        "(see Phase 2 in the plan).")
        else:
            with open(args.adapt_samples) as f:
                adapt_samples = json.load(f)
            logger.info(f"Loaded {len(adapt_samples)} professor adaptation samples")
            n = min(args.n_shot, len(adapt_samples))
            logger.info(f"Adapting OCR to professor: {n} samples, {args.adapt_steps} steps (one-time)")
            ocr.adapt(adapt_samples[:n], steps=args.adapt_steps)

    # Optional local LLM corrector — instantiated lazily, loaded on first use.
    # Failure to load is non-fatal: we log and continue without it.
    # Skipped entirely under --ocr-backend vlm: VLM output is already clean
    # German, and a text-only LLM corrector can only degrade it.
    corrector = None
    if args.llm_correct and not using_vlm:
        try:
            from models.llm_corrector import LLMCorrector
            corrector = LLMCorrector(
                model_id=args.llm_model,
                device=args.device,
                load_in_8bit=args.llm_8bit,
            )
            logger.info(f"LLM corrector configured: {args.llm_model} "
                        f"(load_in_8bit={args.llm_8bit})")
        except Exception as e:
            logger.warning(f"LLM corrector init failed; continuing without it. ({e})")
            corrector = None
    elif using_vlm:
        logger.info("LLM corrector disabled (VLM backend produces clean German).")

    # Collect input/output pairs
    EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    if args.image:
        pairs = [(args.image, args.output or args.image.with_stem(args.image.stem + '_out'))]
    else:
        images = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in EXTS)
        out_dir = args.output_dir or args.image_dir / 'infer_out'
        pairs = [(img, out_dir / img.name) for img in images]
        logger.info(f"Found {len(pairs)} images in {args.image_dir}")

    if not pairs:
        logger.error("No images found to process.")
        sys.exit(1)

    # Process
    all_results = {}
    for img_path, out_path in pairs:
        try:
            results = run_infer(
                image_path=img_path,
                output_path=out_path,
                detector=detector,
                ocr=ocr,
                math_ocr=math_ocr,
                adapt_samples=adapt_samples,
                n_shot=args.n_shot,
                conf_threshold=args.conf,
                annotate_only=args.annotate_only,
                force_tamer=args.force_tamer,
                postprocess_german=not args.no_german_postproc,
                corrector=corrector,
                erase_mode='rectangle' if args.legacy_render else 'ink',
                enable_math_ocr=args.enable_math_ocr,
            )
            all_results[str(img_path)] = results
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    if corrector is not None:
        corrector.unload()
    if using_vlm and hasattr(ocr, 'unload'):
        ocr.unload()

    if args.save_json:
        json_path = out_path.with_suffix('.json') if args.image else (
            (args.output_dir or args.image_dir / 'infer_out') / 'results.json'
        )
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Results saved -> {json_path}")

    logger.info(f"Done. Processed {len(all_results)} image(s).")


if __name__ == '__main__':
    main()
