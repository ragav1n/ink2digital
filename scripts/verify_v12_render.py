"""CPU-only verification of the v12 rendering fixes.

The GPU is down, but the bugs the user reported (tiny font, run-on math,
floating placement, re-typeset header) live entirely in the rendering code,
not the VLM. This replays the *exact* region bboxes + VLM strings captured in
`outputs/jakob_v11/results.json` for pages 7 and 8 through the new rendering
path: the printed-header geometry guard, `is_typeset_text`, the literal-`\\n`
normalizer, and the ink-anchored `render_latex_in_box`.

Output -> outputs/jakob_v12_p78/  (compare against outputs/jakob_v11/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infer import (  # noqa: E402
    _normalize_math_latex,
    is_typeset_text,
    render_latex_in_box,
    render_text_in_box,
)

RESULTS = Path('outputs/jakob_v11/results.json')
OUT = Path('outputs/jakob_v12_p78')
OUT.mkdir(parents=True, exist_ok=True)


def is_printed_header(bbox, ph, pw) -> bool:
    """Same rule added to run_infer's typeset filter."""
    bx1, by1, bx2, by2 = bbox
    wide = (bx2 - bx1) >= 0.20 * pw
    at_top = by1 <= 0.045 * ph
    at_bottom = by2 >= 0.96 * ph
    return wide and (at_top or at_bottom)


def main() -> None:
    data = json.load(open(RESULTS, encoding='utf-8'))
    for key, regions in data.items():
        if 'page-0007' not in key and 'page-0008' not in key:
            continue
        src = Path(key)
        img = Image.open(src).convert('RGB')
        img_np = np.array(img)
        ph, pw = img_np.shape[:2]
        page = src.name
        print(f'\n=== {page}  ({pw}x{ph}) — {len(regions)} regions ===')

        kept = []
        for r in regions:
            x1, y1, x2, y2 = [int(v) for v in r['bbox']]
            if is_printed_header(r['bbox'], ph, pw):
                print(f'  DROP header-band   bbox={r["bbox"]}  {r.get("text","")[:50]!r}')
                continue
            crop = img_np[y1:y2, x1:x2]
            if crop.size and is_typeset_text(crop):
                print(f'  DROP typeset-text  bbox={r["bbox"]}')
                continue
            kept.append(r)

        for r in kept:
            x1, y1, x2, y2 = [int(v) for v in r['bbox']]
            if r.get('type') == 'math':
                latex = _normalize_math_latex(r.get('latex') or r.get('text') or '')
                rows = latex.count('\\\\') + 1
                ok = render_latex_in_box(img, x1, y1, x2, y2, latex)
                print(f'  MATH render={ok}  rows={rows}  bbox={r["bbox"]}')
                print(f'       latex={latex[:110]!r}')
            else:
                text = r.get('text', '').strip()
                if text:
                    render_text_in_box(img, x1, y1, x2, y2, text)
                    print(f'  TEXT bbox={r["bbox"]}  {text[:60]!r}')

        out_path = OUT / page
        img.save(out_path)
        print(f'  -> saved {out_path}')


if __name__ == '__main__':
    main()
