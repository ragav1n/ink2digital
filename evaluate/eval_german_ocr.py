"""
Evaluate German text OCR (TrOCR) on the IAM German test set.

Computes CER, WER with and without post-processing.

Usage:
    python evaluate/eval_german_ocr.py \
        --model microsoft/trocr-large-handwritten \
        --data data/processed/german_text/german_text_test.json \
        --output outputs/eval_german_ocr.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger

from utils.metrics import batch_cer, batch_wer
from utils.german_postprocessing import batch_correct
from utils.image_utils import load_image


def evaluate_trocr(
    model_id: str,
    test_manifest: Path,
    device: str = 'cuda',
    batch_size: int = 8,
    max_new_tokens: int = 128,
    postprocess: bool = True,
    output_path: Path = None,
) -> dict:
    """
    Evaluate a TrOCR model on German test data.

    Args:
        model_id: HuggingFace model ID or local checkpoint path.
        test_manifest: Path to test JSON manifest.
        device: 'cuda' or 'cpu'.
        batch_size: Inference batch size.
        max_new_tokens: Max output tokens.
        postprocess: Apply German post-processing.
        output_path: Where to save results.

    Returns:
        Metrics dict with CER and WER.
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import torch

    device = device if torch.cuda.is_available() else 'cpu'

    logger.info(f"Loading TrOCR: {model_id}")
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
    model.eval()
    logger.info(f"Model loaded on {device}")

    # Load test data
    with open(test_manifest, encoding='utf-8') as f:
        data = json.load(f)
    samples = data.get('samples', [])
    logger.info(f"Evaluating on {len(samples)} test samples")

    hypotheses: List[str] = []
    references: List[str] = []
    errors: int = 0

    # Process in batches
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        pil_images = []
        refs_batch = []

        for sample in batch:
            try:
                img_path = Path(sample['image'])
                if not img_path.is_absolute():
                    img_path = Path('data') / img_path
                img = load_image(img_path, mode='rgb')
                pil_images.append(Image.fromarray(img))
                refs_batch.append(sample['text'])
            except Exception as e:
                logger.debug(f"Skipping sample: {e}")
                errors += 1
                continue

        if not pil_images:
            continue

        # Inference
        try:
            pixel_values = processor(
                images=pil_images, return_tensors='pt'
            ).pixel_values.to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values, max_new_tokens=max_new_tokens
                )
            texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            if postprocess:
                texts = batch_correct(texts, use_spellcheck=False)

            hypotheses.extend(texts)
            references.extend(refs_batch)

        except Exception as e:
            logger.warning(f"Batch {i//batch_size} failed: {e}")
            errors += batch_size

        if (i // batch_size) % 10 == 0:
            logger.info(f"Progress: {min(i+batch_size, len(samples))}/{len(samples)}")

    if not hypotheses:
        logger.error("No predictions generated. Check data and model.")
        return {}

    # Compute metrics
    cer_metrics = batch_cer(hypotheses, references)
    wer_metrics = batch_wer(hypotheses, references)

    metrics = {
        'model': model_id,
        'n_samples': len(hypotheses),
        'n_errors': errors,
        'postprocess': postprocess,
        'CER': cer_metrics['mean_cer'],
        'WER': wer_metrics['mean_wer'],
        'CER_percent': cer_metrics['mean_cer'] * 100,
        'WER_percent': wer_metrics['mean_wer'] * 100,
        # Sample predictions for inspection
        'examples': [
            {'hypothesis': h, 'reference': r, 'cer': c}
            for h, r, c in zip(
                hypotheses[:20], references[:20], cer_metrics['per_sample'][:20]
            )
        ],
    }

    logger.info("\n" + "="*50)
    logger.info("German OCR Evaluation Results")
    logger.info(f"  Model:   {model_id}")
    logger.info(f"  CER:     {metrics['CER']:.4f} ({metrics['CER_percent']:.2f}%)")
    logger.info(f"  WER:     {metrics['WER']:.4f} ({metrics['WER_percent']:.2f}%)")
    logger.info(f"  Samples: {len(hypotheses)} / {len(samples)}")
    logger.info("="*50 + "\n")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved -> {output_path}")

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate German OCR")
    parser.add_argument('--model', type=str, default='microsoft/trocr-large-handwritten')
    parser.add_argument('--data', type=Path,
                        default=Path('data/processed/german_text/german_text_test.json'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--no-postprocess', action='store_true')
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/eval_german_ocr.json'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_trocr(
        model_id=args.model,
        test_manifest=args.data,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        postprocess=not args.no_postprocess,
        output_path=args.output,
    )
