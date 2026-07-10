"""Cross-writer German text OCR evaluation.

Added for the IEEE Access resubmission (Reviewer 1, comment 2): verify that
the deployed text prompt generalises across writers, i.e. that no per-writer
prompt redesign is needed. Runs the UNCHANGED deployed text prompt
(``models/vlm_ocr.py::_TEXT_PROMPT``) zero-shot — no QLoRA adapter — over the
public fhswf/german_handwriting corpus (modern German handwriting, multiple
writers) and reports CER overall and per writer, alongside the lecturer's
zero-shot reference number (5.72%).

Usage (on the RTX 4060 Ti desktop):
    pip install datasets
    python evaluate/eval_crosswriter_german.py \\
        --n-samples 200 --device cuda \\
        --output outputs/eval_crosswriter_german.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger

from utils.device import get_device
from utils.metrics import compute_cer


_TEXT_KEYS = ('text', 'transcription', 'label', 'sentence', 'gt', 'target')
_WRITER_KEYS = ('writer', 'writer_id', 'author', 'writer_name', 'scribe')


def _detect_key(features: dict, candidates: tuple) -> str | None:
    for k in candidates:
        if k in features:
            return k
    return None


def _stratified_indices(writers: list, n: int, seed: int) -> list:
    """Round-robin over writers so each contributes ~equally to the sample."""
    rng = random.Random(seed)
    by_writer = defaultdict(list)
    for i, w in enumerate(writers):
        by_writer[w].append(i)
    for idxs in by_writer.values():
        rng.shuffle(idxs)
    picked, cursors = [], {w: 0 for w in by_writer}
    order = sorted(by_writer)
    while len(picked) < n:
        progressed = False
        for w in order:
            c = cursors[w]
            if c < len(by_writer[w]):
                picked.append(by_writer[w][c])
                cursors[w] = c + 1
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
    return picked


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='fhswf/german_handwriting')
    p.add_argument('--split', default='train')
    p.add_argument('--n-samples', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    p.add_argument('--output', type=Path,
                   default=Path('outputs/eval_crosswriter_german.json'))
    args = p.parse_args()

    device = get_device(args.device)

    from datasets import load_dataset
    logger.info(f"Loading {args.dataset} split={args.split} ...")
    ds = load_dataset(args.dataset, split=args.split)
    logger.info(f"Loaded {len(ds)} rows. Columns: {list(ds.features)}")

    text_key = _detect_key(ds.features, _TEXT_KEYS)
    if text_key is None:
        logger.error(f"No ground-truth column found among {_TEXT_KEYS}; "
                     f"columns are {list(ds.features)}. Pass a corrected key.")
        sys.exit(1)
    writer_key = _detect_key(ds.features, _WRITER_KEYS)
    logger.info(f"Ground truth column: '{text_key}'; "
                f"writer column: {writer_key or '(none found — overall CER only)'}")

    if writer_key:
        writers = [str(w) for w in ds[writer_key]]
        indices = _stratified_indices(writers, args.n_samples, args.seed)
    else:
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(ds)), min(args.n_samples, len(ds)))
    logger.info(f"Sampled {len(indices)} lines (seed={args.seed})")

    # Zero-shot, no adapter: the exact deployed configuration minus the
    # per-writer QLoRA weights, with the UNCHANGED text prompt.
    if device == 'cuda':
        from models.vlm_ocr import VLMOCRBackend
        be = VLMOCRBackend(device=device)
    else:
        from models.vlm_ocr_mlx import MLXVLMOCRBackend
        be = MLXVLMOCRBackend(device=device)

    per_sample, per_writer = [], defaultdict(list)
    for n_done, i in enumerate(indices, 1):
        row = ds[int(i)]
        img = row['image']
        if hasattr(img, 'convert'):
            img = np.array(img.convert('RGB'))
        gt = (row[text_key] or '').strip()
        if not gt:
            continue
        try:
            out = be.predict([img], mode='text', use_adapted=False,
                             postprocess_german=False)
            pred = (out[0] if out else '').strip()
        except Exception as e:
            logger.warning(f"  inference failed on row {i}: {e}")
            pred = ''
        cer = compute_cer(pred, gt)
        writer = str(row[writer_key]) if writer_key else 'unknown'
        per_sample.append({'index': int(i), 'writer': writer, 'gt': gt,
                           'pred': pred, 'cer': cer})
        per_writer[writer].append(cer)
        if n_done % 10 == 0:
            running = sum(s['cer'] for s in per_sample) / len(per_sample)
            logger.info(f"  {n_done}/{len(indices)}  running CER={running*100:.2f}%")

    if hasattr(be, 'unload'):
        be.unload()

    overall = sum(s['cer'] for s in per_sample) / max(len(per_sample), 1)
    writer_stats = {
        w: {'n': len(cers), 'cer': sum(cers) / len(cers)}
        for w, cers in sorted(per_writer.items())
    }
    results = {
        'config': {
            'dataset': args.dataset,
            'split': args.split,
            'n_samples': len(per_sample),
            'seed': args.seed,
            'device': device,
            'prompt': 'models/vlm_ocr.py::_TEXT_PROMPT (unchanged, deployed)',
            'adapter': None,
        },
        'overall_cer': overall,
        'per_writer': writer_stats,
        'per_sample': per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved -> {args.output}")

    print()
    print('=' * 60)
    print('  Cross-writer German OCR (unchanged deployed prompt)')
    print('=' * 60)
    print(f"  overall CER: {overall*100:.2f}%  (n={len(per_sample)})")
    print(f"  lecturer zero-shot reference: 5.72%")
    for w, st in writer_stats.items():
        print(f"  writer {w:<12} n={st['n']:>4}  CER={st['cer']*100:>7.2f}%")
    print('=' * 60)


if __name__ == '__main__':
    main()
