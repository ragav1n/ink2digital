"""YOLO26x fine-tune for the Jakob handwritten-region detector.

YOLO26 is the 2026 release in the Ultralytics line — newer than the YOLOv8x
the project has been using. This script fine-tunes from the COCO-pretrained
yolo26x weights on the v2 Jakob dataset (165 slides; 132 train / 33 val,
classes: text=0, math=1).

Same hi-res settings as ``finetune_jakob_detector_hires.py`` (imgsz=1536) so
we can compare like-for-like against the existing YOLOv8x hires baseline:

    YOLOv8x  hires  -> runs/jakob_detector_v3/hires1536/  (mAP50 ~= 0.54)
    YOLO26x  hires  -> runs/jakob_detector_v3/yolo26x_v1/ (this script)

Training is from the pretrained yolo26x checkpoint rather than continuing a
YOLOv8x checkpoint — architectures differ, weights are not compatible.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        python training/finetune_jakob_detector_yolo26.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='configs/jakob_detection_v2.yaml')
    p.add_argument('--model', type=str, default='yolo26x.pt',
                   help='Pretrained YOLO26 weights (auto-downloads).')
    p.add_argument('--imgsz', type=int, default=1536)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--project', type=str, default='runs/jakob_detector_v3')
    p.add_argument('--name', type=str, default='yolo26x_v1')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    logger.info(f"YOLO26x Jakob detector fine-tune: imgsz={args.imgsz}, "
                f"batch={args.batch}, from {args.model}")
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save_period=10,
        workers=args.workers,
        verbose=True,
        cos_lr=True,
        # Document-friendly augmentation — lecture slides aren't rotated,
        # sheared, flipped or mixed.
        degrees=0.0,
        shear=0.0,
        flipud=0.0,
        fliplr=0.0,
        mixup=0.0,
        copy_paste=0.0,
        translate=0.05,
        scale=0.3,
        mosaic=0.3,
        close_mosaic=10,
        hsv_h=0.0,
        hsv_s=0.3,
        hsv_v=0.3,
        # Optimisation — matches the v8x hires baseline so the comparison is
        # architecture-only.
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
    )

    save_dir = Path(getattr(model.trainer, 'save_dir', args.project))
    best = save_dir / 'weights' / 'best.pt'
    summary = {
        'model': args.model,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'epochs': args.epochs,
        'save_dir': str(save_dir),
        'best_weights': str(best),
        'mAP50': float(getattr(results, 'box', results).map50) if hasattr(results, 'box') else None,
        'mAP50_95': float(getattr(results, 'box', results).map) if hasattr(results, 'box') else None,
    }
    with open(save_dir / 'training_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Done. Best weights: {best}")
    logger.info(f"Summary: {summary}")


if __name__ == '__main__':
    main()
