"""
Quick test: how does pix2tex (LaTeX-OCR) handle Jakob's handwritten math?

Loads the same train/val splits used for TAMER fine-tune, runs pix2tex on each,
prints predictions side-by-side with ground truth and TAMER-Jakob-finetune for comparison.

Pix2tex is trained on PRINTED LaTeX from arXiv — it was not designed for
handwritten input — but its vocab is much larger and Jakob's handwriting
is fairly clean, so it's worth testing before declaring math unsolvable.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import editdistance
import numpy as np
from loguru import logger
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_split(data_root: Path, split: str):
    with open(data_root / split / 'images.pkl', 'rb') as f:
        images = pickle.load(f)
    rows = []
    with open(data_root / split / 'caption.txt') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            rows.append((parts[0], images[parts[0]], parts[1:]))
    return rows


def main():
    from pix2tex.cli import LatexOCR

    data = Path('data/jakob_math_tamer')
    val_rows = load_split(data, 'val')
    train_rows = load_split(data, 'train')
    eval_set = [('val', r) for r in val_rows] + [('train', r) for r in train_rows[:5]]

    logger.info(f"Loading pix2tex...")
    model = LatexOCR()
    logger.info(f"Loaded. Evaluating {len(eval_set)} samples...")

    preds = []
    for split, (fname, img_gray, gt_tokens) in eval_set:
        # pix2tex wants PIL RGB
        if img_gray.ndim == 2:
            arr = np.stack([img_gray] * 3, axis=-1)
        else:
            arr = img_gray
        pil = PILImage.fromarray(arr)
        try:
            latex = model(pil) or ''
        except Exception as e:
            logger.debug(f"  {fname}: {e}")
            latex = ''
        # Re-tokenize pix2tex output the same way our prep script does (whitespace + char-level for digits)
        import re
        tok_re = re.compile(r'\\[A-Za-z]+\*?|\\.|\{|\}|\[|\]|\(|\)|[\^_+\-=<>/.,!|;:\']|[0-9]|[A-Za-z]')
        pred_tokens = tok_re.findall(latex)
        cer = editdistance.eval(pred_tokens, gt_tokens) / max(len(gt_tokens), 1)
        preds.append({
            'split': split, 'fname': fname,
            'gt': ' '.join(gt_tokens),
            'pred_raw': latex,
            'pred_tokens': ' '.join(pred_tokens),
            'token_cer': cer,
            'gt_len': len(gt_tokens),
            'pred_len': len(pred_tokens),
        })

    logger.info("\n=== Per-sample ===")
    for p in preds:
        logger.info(f"\n[{p['split']}] {p['fname']}  (gt_len={p['gt_len']}, pred_len={p['pred_len']}, cer={p['token_cer']:.3f})")
        logger.info(f"  GT:   {p['gt']}")
        logger.info(f"  PRED: {p['pred_raw']}")

    val = [p for p in preds if p['split'] == 'val']
    train = [p for p in preds if p['split'] == 'train']
    logger.info("\n=== SUMMARY (token CER) ===")
    logger.info(f"  pix2tex val:   {np.mean([p['token_cer'] for p in val]):.3f}  (n={len(val)})")
    logger.info(f"  pix2tex train: {np.mean([p['token_cer'] for p in train]):.3f}  (n={len(train)})")
    logger.info("  Reference: Jakob TAMER ft val 6.25 / train 2.83; TAMER orig val 7.26 / train 4.66")

    Path('outputs').mkdir(exist_ok=True)
    with open('outputs/pix2tex_jakob_eval.json', 'w') as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    logger.info("Wrote outputs/pix2tex_jakob_eval.json")


if __name__ == '__main__':
    main()
