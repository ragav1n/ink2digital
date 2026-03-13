"""
Test the baseline pipeline on sample images.

Evaluates the baseline (YOLOv8 + TrOCR + Pix2Tex) and reports:
- Detection metrics (mAP, precision, recall)
- Text OCR metrics (CER, WER)
- Math OCR metrics (BLEU)
- End-to-end accuracy
- Per-image results JSON

Usage:
    python baseline/test_baseline.py \
        --image-dir data/sample_slides \
        --annotations data/sample_annotations.json \
        --output outputs/baseline_test.json

    # Or test on a single image:
    python baseline/test_baseline.py --image path/to/slide.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from loguru import logger


def evaluate_on_annotations(
    pipeline,
    annotations_path: Path,
    output_path: Path,
) -> dict:
    """
    Evaluate pipeline against ground-truth annotations.

    Annotation format:
    {
        "samples": [
            {
                "image": "path/to/image.png",
                "regions": [
                    {"bbox": [x1,y1,x2,y2], "type": "text", "text": "..."},
                    {"bbox": [x1,y1,x2,y2], "type": "math", "text": "$...$"},
                ]
            }
        ]
    }
    """
    from utils.metrics import batch_cer, batch_wer, compute_map, compute_bleu

    with open(annotations_path) as f:
        data = json.load(f)

    all_predictions = []
    all_ground_truths = []
    text_hyps, text_refs = [], []
    math_hyps, math_refs = [], []
    per_image_results = []

    for sample in data.get('samples', []):
        img_path = sample['image']
        gt_regions = sample.get('regions', [])

        t0 = time.time()
        try:
            results = pipeline.process_image(img_path)
        except Exception as e:
            logger.error(f"Pipeline error on {img_path}: {e}")
            continue
        elapsed = time.time() - t0

        # Collect for mAP
        pred_entry = {
            'boxes': [r['bbox'] for r in results],
            'labels': [0 if r['type'] == 'text' else 1 for r in results],
            'scores': [r['confidence'] for r in results],
        }
        gt_entry = {
            'boxes': [r['bbox'] for r in gt_regions],
            'labels': [0 if r['type'] == 'text' else 1 for r in gt_regions],
        }
        all_predictions.append(pred_entry)
        all_ground_truths.append(gt_entry)

        # Match predictions to ground truth by IoU for OCR eval
        from utils.metrics import compute_iou
        for pred in results:
            best_iou = 0.0
            best_gt = None
            for gt in gt_regions:
                if gt['type'] != pred['type']:
                    continue
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_gt and best_iou > 0.5:
                if pred['type'] == 'text':
                    text_hyps.append(pred['text'])
                    text_refs.append(best_gt['text'])
                else:
                    math_hyps.append(pred['text'])
                    math_refs.append(best_gt['text'])

        per_image_results.append({
            'image': img_path,
            'predictions': results,
            'ground_truth': gt_regions,
            'inference_time': elapsed,
        })

    # Compute metrics
    metrics = {}

    if all_predictions:
        map_result = compute_map(all_predictions, all_ground_truths,
                                  class_names=['text', 'math'])
        metrics['detection'] = map_result

    if text_hyps:
        metrics['text_ocr'] = batch_cer(text_hyps, text_refs)
        metrics['text_wer'] = batch_wer(text_hyps, text_refs)

    if math_hyps:
        bleu_scores = [compute_bleu(h, r) for h, r in zip(math_hyps, math_refs)]
        metrics['math_bleu'] = float(np.mean(bleu_scores))

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'per_image': per_image_results,
            'n_images': len(per_image_results),
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved -> {output_path}")
    return metrics


def test_single_image(pipeline, image_path: str, output_path: Path) -> None:
    """Run pipeline on a single image and show results."""
    logger.info(f"Processing: {image_path}")
    t0 = time.time()
    results = pipeline.process_image(image_path)
    elapsed = time.time() - t0

    logger.info(f"Found {len(results)} regions in {elapsed:.2f}s")
    for i, r in enumerate(results):
        logger.info(f"  Region {i+1}: [{r['type']}] conf={r['confidence']:.2f} | {r['text'][:60]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'image': image_path,
            'results': results,
            'inference_time': elapsed,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved -> {output_path}")


def print_metrics_summary(metrics: dict) -> None:
    """Print a formatted metrics summary."""
    print("\n" + "="*60)
    print("BASELINE EVALUATION RESULTS")
    print("="*60)

    if 'detection' in metrics:
        det = metrics['detection']
        print(f"\nDetection (mAP@{det['iou_threshold']}):")
        print(f"  Overall mAP: {det['mAP']:.4f} ({det['mAP']*100:.1f}%)")
        for cls, ap in det.get('per_class_AP', {}).items():
            print(f"  AP[{cls}]: {ap:.4f}")

    if 'text_ocr' in metrics:
        print(f"\nText OCR:")
        print(f"  CER: {metrics['text_ocr']['mean_cer']:.4f} ({metrics['text_ocr']['mean_cer']*100:.1f}%)")
        if 'text_wer' in metrics:
            print(f"  WER: {metrics['text_wer']['mean_wer']:.4f} ({metrics['text_wer']['mean_wer']*100:.1f}%)")

    if 'math_bleu' in metrics:
        print(f"\nMath OCR:")
        print(f"  BLEU: {metrics['math_bleu']:.4f}")

    print("="*60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Test baseline pipeline")
    parser.add_argument('--image', type=str, default=None,
                        help='Single image path for quick test')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Directory of slide images for batch evaluation')
    parser.add_argument('--annotations', type=str, default=None,
                        help='JSON file with ground-truth annotations')
    parser.add_argument('--output', type=Path, default=Path('outputs/baseline_test.json'),
                        help='Output JSON file for results')
    parser.add_argument('--detector-weights', type=str, default='yolov8x.pt',
                        help='YOLOv8 weights file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-postprocess', action='store_true',
                        help='Disable German post-processing')
    return parser.parse_args()


def main():
    args = parse_args()

    # Import and initialize pipeline
    logger.info("Initializing baseline pipeline...")
    from baseline.baseline_pipeline import BaselinePipeline

    pipeline = BaselinePipeline(
        detector_weights=args.detector_weights,
        device=args.device,
        postprocess_german=not args.no_postprocess,
    )

    if args.image:
        # Single image test
        test_single_image(pipeline, args.image, args.output)

    elif args.annotations:
        # Full evaluation with ground truth
        metrics = evaluate_on_annotations(
            pipeline,
            Path(args.annotations),
            args.output,
        )
        print_metrics_summary(metrics)

    elif args.image_dir:
        # Batch inference without ground truth
        img_dir = Path(args.image_dir)
        images = sorted(img_dir.glob('*.png')) + sorted(img_dir.glob('*.jpg'))
        logger.info(f"Found {len(images)} images in {img_dir}")

        all_results = pipeline.process_batch(images)
        output_data = [
            {'image': str(img), 'results': res}
            for img, res in zip(images, all_results)
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Batch results saved -> {args.output}")

    else:
        logger.warning("No input specified. Use --image, --image-dir, or --annotations.")
        logger.info("Running pipeline self-test with a synthetic image...")

        # Self-test with blank image
        from utils.image_utils import numpy_to_pil
        import numpy as np
        test_img = np.ones((720, 1280, 3), dtype=np.uint8) * 240
        results = pipeline.process_image(test_img)
        logger.info(f"Self-test complete: {len(results)} regions detected on blank image")


if __name__ == '__main__':
    main()
