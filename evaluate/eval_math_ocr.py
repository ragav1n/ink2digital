"""
Evaluate math OCR (Pix2Tex / TAMER) on CROHME test data.

Computes BLEU score against LaTeX ground truth.

Usage:
    python evaluate/eval_math_ocr.py \
        --model pix2tex \
        --data data/processed/math/math_test.json \
        --output outputs/eval_math_ocr.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import List

import numpy as np
from loguru import logger

from utils.metrics import compute_bleu
from utils.image_utils import load_image


def evaluate_math_ocr(
    model_name: str,
    test_manifest: Path,
    output_path: Path = None,
) -> dict:
    """
    Evaluate a math OCR model.

    Supports:
        - 'pix2tex': Pix2Tex LaTeX OCR
        - 'tamer': TAMER (requires TAMER repo)

    Returns metrics dict with BLEU score.
    """
    with open(test_manifest, encoding='utf-8') as f:
        data = json.load(f)
    samples = data.get('samples', [])
    logger.info(f"Evaluating {model_name} on {len(samples)} samples")

    # Load model
    if model_name == 'pix2tex':
        recognizer = _load_pix2tex()
    elif model_name == 'tamer':
        recognizer = _load_tamer()
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'pix2tex' or 'tamer'")

    hypotheses: List[str] = []
    references: List[str] = []

    for sample in samples:
        img_path = Path(sample.get('image', ''))
        if not img_path.exists():
            # Try relative to data/
            img_path = Path('data') / img_path
        if not img_path.exists():
            continue

        try:
            img = load_image(img_path, mode='rgb')
            latex = recognizer(img)
            hypotheses.append(latex)
            references.append(sample['latex'])
        except Exception as e:
            logger.debug(f"Error on {img_path}: {e}")

    if not hypotheses:
        logger.error("No predictions generated.")
        return {}

    # BLEU scores
    bleu_scores = [compute_bleu(h, r) for h, r in zip(hypotheses, references)]
    mean_bleu = float(np.mean(bleu_scores))

    metrics = {
        'model': model_name,
        'n_samples': len(hypotheses),
        'BLEU': mean_bleu,
        'BLEU_percent': mean_bleu * 100,
        'examples': [
            {'hypothesis': h, 'reference': r, 'bleu': b}
            for h, r, b in zip(hypotheses[:10], references[:10], bleu_scores[:10])
        ],
    }

    logger.info("\n" + "="*50)
    logger.info(f"Math OCR Evaluation ({model_name})")
    logger.info(f"  BLEU: {mean_bleu:.4f} ({mean_bleu*100:.1f}%)")
    logger.info(f"  Samples: {len(hypotheses)}")
    logger.info("="*50)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved -> {output_path}")

    return metrics


def _load_pix2tex():
    """Load Pix2Tex and return a callable (image -> latex string)."""
    try:
        from pix2tex.cli import LatexOCR
        from PIL import Image
        model = LatexOCR()

        def recognize(img_np):
            from utils.image_utils import numpy_to_pil
            pil = numpy_to_pil(img_np)
            return model(pil) or ''

        return recognize
    except ImportError:
        logger.error("pix2tex not installed. Install: pip install pix2tex")
        raise


def _load_tamer():
    """Load TAMER model and return callable."""
    try:
        import sys
        sys.path.insert(0, str(Path('TAMER').resolve()))
        from tamer.model.tamer import TAMER as TAMERModel
        # Use the latest checkpoint
        from pathlib import Path as P
        ckpts = sorted(P('TAMER/lightning_logs').rglob('*.ckpt'))
        if not ckpts:
            raise FileNotFoundError("No TAMER checkpoints found in TAMER/lightning_logs")
        ckpt = str(ckpts[-1])
        logger.info(f"Loading TAMER checkpoint: {ckpt}")
        model = TAMERModel.load_from_checkpoint(ckpt)
        model.eval()

        def recognize(img_np):
            # TAMER expects specific input format — placeholder
            # Full TAMER inference will be implemented in Phase 2
            return ''

        return recognize
    except Exception as e:
        logger.error(f"TAMER load failed: {e}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate math OCR")
    parser.add_argument('--model', type=str, default='pix2tex', choices=['pix2tex', 'tamer'])
    parser.add_argument('--data', type=Path, default=Path('data/processed/math/math_test.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/eval_math_ocr.json'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_math_ocr(args.model, args.data, args.output)
