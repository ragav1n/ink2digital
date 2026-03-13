"""
Evaluate a trained YOLOv8 detection model.

Computes mAP, precision, recall for text/math region detection.

Usage:
    python evaluate/eval_detection.py \
        --model runs/detect/baseline_v1/weights/best.pt \
        --data configs/handwriting_detection.yaml \
        --output outputs/eval_detection.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger


def evaluate_detector(
    model_path: str,
    data_yaml: str,
    device: str = '0',
    split: str = 'test',
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    output_path: Path = None,
) -> dict:
    """
    Evaluate a YOLOv8 model on a dataset split.

    Returns dict with mAP, precision, recall, and per-class metrics.
    """
    from ultralytics import YOLO

    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)

    logger.info(f"Evaluating on '{split}' split...")
    results = model.val(
        data=data_yaml,
        split=split,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=True,
    )

    # Extract metrics
    metrics = {}
    if hasattr(results, 'box'):
        box = results.box
        metrics = {
            'mAP50': float(box.map50),
            'mAP50_95': float(box.map),
            'precision': float(box.mp),
            'recall': float(box.mr),
            'per_class': {},
        }
        if hasattr(box, 'ap_class_index') and box.ap_class_index is not None:
            names = model.names
            for i, cls_idx in enumerate(box.ap_class_index):
                cls_name = names.get(int(cls_idx), str(cls_idx))
                metrics['per_class'][cls_name] = {
                    'AP50': float(box.ap50[i]) if hasattr(box, 'ap50') else None,
                    'AP50_95': float(box.ap[i]) if hasattr(box, 'ap') else None,
                }

    metrics['model'] = model_path
    metrics['data'] = data_yaml
    metrics['split'] = split
    metrics['conf_threshold'] = conf
    metrics['iou_threshold'] = iou

    # Print summary
    logger.info("\n" + "="*50)
    logger.info(f"Detection Evaluation Results")
    logger.info(f"  Model:    {model_path}")
    logger.info(f"  Split:    {split}")
    logger.info(f"  mAP@0.5:      {metrics.get('mAP50', 'N/A'):.4f}")
    logger.info(f"  mAP@0.5:0.95: {metrics.get('mAP50_95', 'N/A'):.4f}")
    logger.info(f"  Precision:    {metrics.get('precision', 'N/A'):.4f}")
    logger.info(f"  Recall:       {metrics.get('recall', 'N/A'):.4f}")
    for cls, vals in metrics.get('per_class', {}).items():
        logger.info(f"  AP[{cls}]: {vals.get('AP50', 'N/A'):.4f}")
    logger.info("="*50 + "\n")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved -> {output_path}")

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate detection model")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, default='configs/handwriting_detection.yaml')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--output', type=Path, default=Path('outputs/eval_detection.json'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_detector(
        model_path=args.model,
        data_yaml=args.data,
        device=args.device,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        output_path=args.output,
    )
