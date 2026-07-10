"""
Side-by-side eval: original TAMER vs Jakob-fine-tuned TAMER on Jakob math crops.

Loads ground-truth tokenized captions, runs both models, computes token-level
edit distance + Exact Match. Reports per-sample diffs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import editdistance
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_split(data_root: Path, split: str):
    """Return list of (fname, image_np, gt_tokens)."""
    with open(data_root / split / 'images.pkl', 'rb') as f:
        images = pickle.load(f)
    rows = []
    with open(data_root / split / 'caption.txt') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            fname = parts[0]
            tokens = parts[1:]
            rows.append((fname, images[fname], tokens))
    return rows


def gray_to_rgb(arr):
    """TAMERMathOCR.recognize expects an RGB ndarray."""
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    return arr


def token_cer(pred_tokens, ref_tokens):
    """Token-level edit distance / len(ref). 0 = perfect, 1 = totally wrong."""
    if not ref_tokens:
        return 1.0 if pred_tokens else 0.0
    return editdistance.eval(pred_tokens, ref_tokens) / len(ref_tokens)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/jakob_math_tamer'))
    p.add_argument('--original', type=Path, default=Path('TAMER/lightning_logs/version_3'))
    p.add_argument('--finetuned', type=Path,
                   default=Path('checkpoint/tamer_jakob/lightning_logs/jakob'))
    p.add_argument('--output', type=Path,
                   default=Path('outputs/tamer_jakob_eval.json'))
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    from models.math_ocr_tamer import TAMERMathOCR

    val_rows = load_split(args.data, 'val')
    train_rows = load_split(args.data, 'train')
    # Also evaluate on a few train samples — over-fit signal
    eval_set = [('val', r) for r in val_rows] + [('train', r) for r in train_rows[:3]]

    results = {}
    for tag, ckpt_dir in [('original_v3', args.original),
                          ('jakob_finetune', args.finetuned)]:
        logger.info(f"=== {tag} ===")
        model = TAMERMathOCR(checkpoint_dir=str(ckpt_dir), fallback_to_pix2tex=False)
        if model.model is None:
            logger.error(f"{tag} failed to load")
            continue

        preds = []
        for split, (fname, img_gray, gt_tokens) in eval_set:
            img_rgb = gray_to_rgb(img_gray)
            try:
                latex = model.recognize(img_rgb)
            except Exception as e:
                logger.debug(f"  {fname} failed: {e}")
                latex = ''
            pred_tokens = latex.split()
            cer = token_cer(pred_tokens, gt_tokens)
            preds.append({
                'split': split, 'fname': fname,
                'gt': ' '.join(gt_tokens),
                'pred': latex,
                'gt_len': len(gt_tokens),
                'token_cer': cer,
                'exact_match': pred_tokens == gt_tokens,
            })
        results[tag] = {
            'mean_token_cer': float(np.mean([p['token_cer'] for p in preds])),
            'val_mean_cer': float(np.mean([p['token_cer'] for p in preds if p['split']=='val'])),
            'train_mean_cer': float(np.mean([p['token_cer'] for p in preds if p['split']=='train'])),
            'exact_match_count': sum(p['exact_match'] for p in preds),
            'predictions': preds,
        }
        logger.info(f"  Mean token CER: {results[tag]['mean_token_cer']:.3f}  "
                    f"(val: {results[tag]['val_mean_cer']:.3f}, "
                    f"train: {results[tag]['train_mean_cer']:.3f})")

        del model
        import torch
        torch.cuda.empty_cache()

    # Per-sample side-by-side
    logger.info("\n=== Per-sample comparison ===")
    by_fname = {}
    for tag, r in results.items():
        for p in r['predictions']:
            by_fname.setdefault(p['fname'], {'split': p['split'], 'gt': p['gt']})[tag] = p

    for fname, e in by_fname.items():
        logger.info(f"\n[{e['split']}] {fname}")
        logger.info(f"  GT     : {e['gt']}")
        for tag in ('original_v3', 'jakob_finetune'):
            if tag in e:
                logger.info(f"  {tag:14}: cer={e[tag]['token_cer']:.3f}  {e[tag]['pred']}")

    logger.info("\n=== SUMMARY ===")
    if 'original_v3' in results and 'jakob_finetune' in results:
        o = results['original_v3']['val_mean_cer']
        f = results['jakob_finetune']['val_mean_cer']
        logger.info(f"  Val token CER:  orig {o:.3f}  ->  finetuned {f:.3f}  ({'WIN' if f<o else 'LOSS'} by {(o-f)*100:+.1f}pp)")
        o2 = results['original_v3']['train_mean_cer']
        f2 = results['jakob_finetune']['train_mean_cer']
        logger.info(f"  Train CER (overfit signal): orig {o2:.3f} -> finetuned {f2:.3f}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {args.output}")


if __name__ == '__main__':
    main()
