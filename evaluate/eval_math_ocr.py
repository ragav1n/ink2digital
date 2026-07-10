"""Quantitative math-OCR evaluation across all available backends.

Reads a manifest of `{image, gt_latex}` entries, runs one or more math OCR
backends (VLM / TAMER / pix2tex / TrOCR-FT) on every crop, and writes a JSON
report with ExpRate / CER / BLEU — overall, per auto-derived tag, and
per-sample. A short table is also printed to stdout.

Each prediction is scored on two tracks:
- ``raw``  — the model's untouched output. Measures the OCR model.
- ``post`` — after ``is_garbage_latex`` + ``_normalize_math_latex`` (same
  pipeline ``infer.py`` applies before rendering). Measures what ends up on
  the rendered slide.

Tags are derived from ``gt_latex`` so they are mechanical and reproducible:
``with-matrix``, ``with-integral``, ``with-sum``, ``with-sqrt``, ``with-frac``,
``multi-line``, and a length bucket (``short`` / ``medium`` / ``long``).

Usage:
    python evaluate/eval_math_ocr.py \\
        --data data/jakob_math_eval/samples.json \\
        --data data/jakob_math/transcriptions.json \\
        --backend all --device auto \\
        --output outputs/eval_math_ocr_jakob.json
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from utils.device import get_device
from utils.image_utils import load_image
from utils.metrics import _edit_distance, compute_bleu, compute_cer
from infer import _normalize_math_latex, is_garbage_latex


# ---------------------------------------------------------------------------
# Canonicalization & tag derivation
# ---------------------------------------------------------------------------

_FORCED_SPACE_RE = re.compile(r'\\[,;:!]|\\ |~')
_WS_RE = re.compile(r'\s+')
_ENV_BLOCK_RE = re.compile(r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', re.DOTALL)
_MATRIX_ENV_RE = re.compile(r'\\begin\{(?:p|b|v|V|B|small)?matrix\}|\\begin\{cases\}')


def canonicalize_latex(s: str) -> str:
    """Normalize a LaTeX string for exact-match comparison (ExpRate).

    Case is preserved — math is case-sensitive ($x$ != $X$). The goal is to
    fold away differences that don't change the rendered expression: outer
    math delimiters, stray whitespace, forced-spacing commands, and brace
    padding.
    """
    if not s:
        return ''
    expr = s.strip()
    # Strip outer math delimiters (the model sometimes emits $ despite prompt).
    for opener, closer in (('$$', '$$'), ('$', '$'), ('\\(', '\\)'), ('\\[', '\\]')):
        if expr.startswith(opener) and expr.endswith(closer) and len(expr) > len(opener) + len(closer):
            expr = expr[len(opener):-len(closer)].strip()
            break
    # Drop any remaining stray $.
    expr = expr.replace('$', '')
    # Forced-spacing commands ( \, \; \: \! \  ~ ) become a single space.
    expr = _FORCED_SPACE_RE.sub(' ', expr)
    # Collapse whitespace runs.
    expr = _WS_RE.sub(' ', expr).strip()
    # Trim whitespace just inside braces: "{ x }" -> "{x}".
    expr = re.sub(r'\{\s+', '{', expr)
    expr = re.sub(r'\s+\}', '}', expr)
    return expr


def tag_expr(gt_latex: str) -> List[str]:
    """Return the set of tags applicable to a ground-truth expression."""
    tags: List[str] = []
    if not gt_latex:
        return tags
    if _MATRIX_ENV_RE.search(gt_latex):
        tags.append('with-matrix')
    if '\\int' in gt_latex:
        tags.append('with-integral')
    if '\\sum' in gt_latex:
        tags.append('with-sum')
    if '\\sqrt' in gt_latex:
        tags.append('with-sqrt')
    if '\\frac' in gt_latex:
        tags.append('with-frac')
    # multi-line: any \\ outside a \begin{...}\end{...} block.
    outside = _ENV_BLOCK_RE.sub('', gt_latex)
    if '\\\\' in outside:
        tags.append('multi-line')
    n = len(canonicalize_latex(gt_latex))
    if n < 30:
        tags.append('short')
    elif n <= 80:
        tags.append('medium')
    else:
        tags.append('long')
    return tags


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------

def load_samples(data_paths: List[Path]) -> List[Dict]:
    """Concatenate one or more manifests. Accepts a top-level list or a
    ``{"samples": [...]}`` wrap. Each entry must have an ``image`` key plus
    one of ``gt_latex`` / ``latex`` / ``text`` for the ground truth."""
    samples: List[Dict] = []
    for p in data_paths:
        with open(p, encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get('samples', [])
        for entry in data:
            gt = entry.get('gt_latex') or entry.get('latex') or entry.get('text') or ''
            samples.append({
                'image': entry['image'],
                'gt_latex': gt,
                'source_file': str(p),
            })
    return samples


def _resolve_image_path(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    cand = Path('data') / p
    if cand.exists():
        return cand
    return p  # will fail downstream with a clear error


# ---------------------------------------------------------------------------
# Backend loading
# ---------------------------------------------------------------------------

def load_backend(name: str, device: str) -> Tuple[Callable[[any], str], Callable[[], None]]:
    """Return ``(infer_fn, unload_fn)`` for the named backend."""
    if name == 'vlm':
        if device == 'cuda':
            from models.vlm_ocr import VLMOCRBackend
            be = VLMOCRBackend(device=device)
        else:
            from models.vlm_ocr_mlx import MLXVLMOCRBackend
            be = MLXVLMOCRBackend(device=device)

        def infer(img):
            out = be.predict([img], mode='math')
            return out[0] if out else ''

        def unload():
            if hasattr(be, 'unload'):
                be.unload()

        return infer, unload

    if name == 'tamer':
        from models.math_ocr_tamer import TAMERMathOCR
        be = TAMERMathOCR(device=device, prefer_pix2tex=False, fallback_to_pix2tex=False)

        def infer(img):
            return be.recognize(img)

        return infer, lambda: None

    if name == 'pix2tex':
        from models.math_ocr_tamer import TAMERMathOCR
        be = TAMERMathOCR(device=device, prefer_pix2tex=True)

        def infer(img):
            return be.recognize(img)

        return infer, lambda: None

    if name == 'trocr':
        from models.meta_learning_ocr import MAMLOCRWrapper
        be = MAMLOCRWrapper(base_model_path='checkpoint/trocr_german/best', device=device)

        def infer(img):
            out = be.predict([img], use_adapted=False, postprocess_german=False)
            return out[0] if out else ''

        return infer, lambda: None

    if name == 'paddleocr-vl':
        from models.paddleocr_vl_ocr import PaddleOCRVLBackend
        be = PaddleOCRVLBackend(device=device)

        def infer(img):
            out = be.predict([img], mode='math')
            return out[0] if out else ''

        def unload():
            if hasattr(be, 'unload'):
                be.unload()

        return infer, unload

    if name == 'smoldocling':
        from models.smoldocling_ocr import SmolDoclingBackend
        be = SmolDoclingBackend(device=device)

        def infer(img):
            out = be.predict([img], mode='math')
            return out[0] if out else ''

        def unload():
            if hasattr(be, 'unload'):
                be.unload()

        return infer, unload

    raise ValueError(f"Unknown backend: {name}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _safe_cer(hyp: str, ref: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return compute_cer(hyp, ref)


def _score_pair(hyp: str, ref: str) -> Dict:
    h_can = canonicalize_latex(hyp)
    r_can = canonicalize_latex(ref)
    return {
        'exp_rate': 1 if (h_can != '' and h_can == r_can) else 0,
        'cer': _safe_cer(hyp, ref),
        'bleu': compute_bleu(hyp, ref),
        'edit': _edit_distance(hyp, ref),
    }


def _aggregate(per_sample: List[Dict], track: str) -> Dict:
    n = len(per_sample)
    if n == 0:
        return {'exp_rate': 0.0, 'cer': 0.0, 'bleu': 0.0, 'n': 0}
    return {
        'exp_rate': sum(s[track]['exp_rate'] for s in per_sample) / n,
        'cer':      sum(s[track]['cer']      for s in per_sample) / n,
        'bleu':     sum(s[track]['bleu']     for s in per_sample) / n,
        'n':        n,
    }


def _per_tag(per_sample: List[Dict], tracks: List[str]) -> Dict:
    all_tags: set = set()
    for s in per_sample:
        all_tags.update(s['tags'])
    out: Dict[str, Dict] = {}
    for t in sorted(all_tags):
        subset = [s for s in per_sample if t in s['tags']]
        cell: Dict = {'n': len(subset)}
        for track in tracks:
            cell[track] = _aggregate(subset, track)
            cell[track].pop('n', None)
        out[t] = cell
    return out


# ---------------------------------------------------------------------------
# Per-backend driver
# ---------------------------------------------------------------------------

def run_backend(name: str, samples: List[Dict], device: str, include_post: bool) -> Dict:
    logger.info(f"Loading backend '{name}' on {device}...")
    infer_fn, unload = load_backend(name, device)

    per_sample: List[Dict] = []
    t0 = time.time()

    for i, s in enumerate(samples):
        img_path = _resolve_image_path(s['image'])
        if not img_path.exists():
            logger.warning(f"  missing image: {img_path}")
            continue
        try:
            img = load_image(img_path, mode='rgb')
        except Exception as e:
            logger.warning(f"  load failed {img_path}: {e}")
            continue

        try:
            raw = infer_fn(img) or ''
        except Exception as e:
            logger.warning(f"  inference failed on {img_path}: {e}")
            raw = ''

        post = '' if is_garbage_latex(raw) else _normalize_math_latex(raw)

        gt = s['gt_latex']
        entry = {
            'image': s['image'],
            'gt_latex': gt,
            'pred_raw': raw,
            'pred_post': post,
            'tags': tag_expr(gt),
            'raw': _score_pair(raw, gt),
        }
        if include_post:
            entry['post'] = _score_pair(post, gt)
        per_sample.append(entry)

        if (i + 1) % 10 == 0:
            logger.info(f"  [{name}] {i + 1}/{len(samples)} processed")

    tracks = ['raw'] + (['post'] if include_post else [])
    overall = {t: _aggregate(per_sample, t) for t in tracks}
    per_tag = _per_tag(per_sample, tracks)
    dt = time.time() - t0

    logger.info(
        f"[{name}] done in {dt:.1f}s  "
        f"raw ExpRate={overall['raw']['exp_rate']*100:.1f}%  "
        f"raw CER={overall['raw']['cer']*100:.1f}%"
        + (f"  post ExpRate={overall['post']['exp_rate']*100:.1f}%" if include_post else '')
    )

    try:
        unload()
    except Exception as e:
        logger.debug(f"unload failed: {e}")
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        'overall': overall,
        'per_tag': per_tag,
        'per_sample': per_sample,
        'wall_seconds': round(dt, 1),
    }


# ---------------------------------------------------------------------------
# Stdout summary table
# ---------------------------------------------------------------------------

def _print_summary(results: Dict, include_post: bool) -> None:
    backends = results['backends']
    tracks = ['raw'] + (['post'] if include_post else [])

    print()
    print("=" * 78)
    print("  Math OCR Evaluation — Summary")
    print("=" * 78)

    for track in tracks:
        print(f"\n  Track: {track}")
        print(f"  {'backend':<10}  {'n':>4}  {'ExpRate':>9}  {'CER':>9}  {'BLEU':>9}")
        for name, be in backends.items():
            if 'error' in be:
                print(f"  {name:<10}  ERROR: {be['error']}")
                continue
            ov = be['overall'][track]
            print(f"  {name:<10}  {ov['n']:>4}  "
                  f"{ov['exp_rate']*100:>8.2f}%  "
                  f"{ov['cer']*100:>8.2f}%  "
                  f"{ov['bleu']*100:>8.2f}%")

    print(f"\n  Per-tag (track=raw)  — cells: ExpRate% / CER% (n)")
    tag_names: set = set()
    for be in backends.values():
        if 'per_tag' in be:
            tag_names.update(be['per_tag'].keys())
    header = f"  {'tag':<14}"
    for name in backends:
        header += f"  {name:>16}"
    print(header)
    for tag in sorted(tag_names):
        row = f"  {tag:<14}"
        for name, be in backends.items():
            cell = be.get('per_tag', {}).get(tag)
            if not cell:
                row += f"  {'-':>16}"
                continue
            ov = cell['raw']
            n = cell['n']
            row += f"  {ov['exp_rate']*100:>5.1f}/{ov['cer']*100:>5.1f}({n:>3})"
        print(row)
    print()
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, action='append', required=True,
                   help='Manifest JSON (repeatable; manifests are concatenated).')
    p.add_argument('--backend', default='vlm',
                   choices=['vlm', 'tamer', 'pix2tex', 'trocr', 'paddleocr-vl',
                            'smoldocling', 'all'],
                   help='Which OCR backend(s) to evaluate.')
    p.add_argument('--device', default='auto',
                   choices=['auto', 'cuda', 'mps', 'cpu'])
    p.add_argument('--output', type=Path,
                   default=Path('outputs/eval_math_ocr.json'))
    p.add_argument('--raw-only', action='store_true',
                   help='Skip the post-processed metric track.')
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    include_post = not args.raw_only

    samples = load_samples(args.data)
    if not samples:
        logger.error("No samples loaded — check --data path(s).")
        sys.exit(1)
    logger.info(f"Loaded {len(samples)} samples from {len(args.data)} manifest(s)")
    logger.info(f"Device: {device}")

    backends_to_run = (['vlm', 'tamer', 'pix2tex', 'trocr', 'paddleocr-vl', 'smoldocling']
                       if args.backend == 'all' else [args.backend])

    results = {
        'config': {
            'data_files': [str(p) for p in args.data],
            'n_samples': len(samples),
            'backends_run': backends_to_run,
            'device': device,
            'git_sha': _git_sha(),
            'include_post': include_post,
        },
        'backends': {},
    }

    for name in backends_to_run:
        try:
            results['backends'][name] = run_backend(name, samples, device, include_post)
        except Exception as e:
            logger.error(f"Backend '{name}' failed: {e}")
            results['backends'][name] = {'error': str(e)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved -> {args.output}")

    _print_summary(results, include_post)


if __name__ == '__main__':
    main()
