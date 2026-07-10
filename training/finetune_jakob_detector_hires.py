"""
High-resolution YOLOv8 fine-tune for the Jakob handwritten-region detector.

Why this script exists:
- The v1/v2 detectors and all the "hires" runs were trained at imgsz=960. The
  Jakob slides are 4000x2250, so a typical handwritten line box shrinks to
  ~25-80 px at 960 — too small for YOLO to localise reliably. All five existing
  runs plateau at mAP50 ~= 0.40-0.46, which shows up downstream as fragmented /
  truncated detections (one equation line split into several boxes).
- Nobody has actually trained at high resolution yet ("hires" was a misnomer).
  This run does: imgsz=1536, which roughly triples the on-grid box size.
- Augmentation is also fixed for fixed-layout documents: no rotation, shear,
  flips, mixup or copy-paste (those distort a lecture slide); only mild
  translate / scale / brightness and a light mosaic for small-object variety.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python training/finetune_jakob_detector_hires.py
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
    p.add_argument('--model', type=str,
                   default='runs/detect/runs/detect/jakob_finetune_v2_hires_w0/weights/best.pt',
                   help='Checkpoint to continue from (best existing Jakob detector).')
    p.add_argument('--imgsz', type=int, default=1536)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--project', type=str, default='runs/jakob_detector_v3')
    p.add_argument('--name', type=str, default='hires1536')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    logger.info(f"Hi-res Jakob detector fine-tune: imgsz={args.imgsz}, "
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
        # Document-appropriate augmentation — a lecture slide is never rotated,
        # sheared, flipped or mixed with another slide.
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
        # Optimisation
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
    )

    save_dir = Path(getattr(model.trainer, 'save_dir', args.project))
    best = save_dir / 'weights' / 'best.pt'
    summary = {
        'imgsz': args.imgsz,
        'batch': args.batch,
        'epochs': args.epochs,
        'from_checkpoint': args.model,
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
