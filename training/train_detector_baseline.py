"""
Train YOLOv8 detector for handwritten text/math region detection.
Phase 1, Week 3.

This script fine-tunes YOLOv8x on DocLayNet data (prepared by
scripts/prepare_doclaynet.py) to detect two classes:
  0: text (handwritten text regions)
  1: math (mathematical expression regions)

Expected training time: 6-8 hours on RTX 4060 Ti 16GB
Expected results: mAP50 88-92%, Precision 90%+

Usage:
    python training/train_detector_baseline.py \
        --data configs/handwriting_detection.yaml \
        --epochs 100 \
        --batch 16 \
        --device 0

    # Resume training:
    python training/train_detector_baseline.py \
        --resume runs/detect/baseline_v1/weights/last.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from loguru import logger


def setup_training_environment():
    """Set CUDA memory management for RTX 4060 Ti 16GB."""
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')


def validate_dataset(data_yaml: Path) -> bool:
    """
    Validate that the dataset YAML and image directories exist.
    Returns True if valid, False if dataset needs to be prepared first.
    """
    import yaml
    if not data_yaml.exists():
        logger.error(f"Dataset config not found: {data_yaml}")
        return False

    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    base_path = Path(config.get('path', '.'))
    for split in ['train', 'val']:
        split_path = base_path / config.get(split, split)
        if not split_path.exists():
            logger.error(f"Dataset split not found: {split_path}")
            logger.error("Run: python scripts/prepare_doclaynet.py first")
            return False

        n_images = len(list(split_path.glob('*.png')) + list(split_path.glob('*.jpg')))
        logger.info(f"  {split}: {n_images} images")

    return True


def train_yolov8(
    data_yaml: str = 'configs/handwriting_detection.yaml',
    model_size: str = 'yolov8x',
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: str = '0',
    project: str = 'runs/detect',
    name: str = 'baseline_v1',
    pretrained: bool = True,
    resume: str = None,
    patience: int = 20,
    save_period: int = 10,
    workers: int = 8,
) -> dict:
    """
    Train YOLOv8 on the handwriting detection dataset.

    Args:
        data_yaml: Path to dataset YAML config.
        model_size: YOLOv8 model variant ('yolov8n', 'yolov8s', ..., 'yolov8x').
        epochs: Number of training epochs.
        batch: Batch size (reduce to 8 if OOM with 16GB).
        imgsz: Training image size.
        device: CUDA device ID ('0', '0,1', 'cpu').
        project: Output directory prefix.
        name: Run name (results saved to project/name/).
        pretrained: Use COCO pretrained weights.
        resume: Path to resume from (overrides other weight options).
        patience: Early stopping patience (0 to disable).
        save_period: Save checkpoint every N epochs.
        workers: DataLoader worker processes.

    Returns:
        dict with training results and best checkpoint path.
    """
    from ultralytics import YOLO

    setup_training_environment()

    # Load model
    if resume:
        logger.info(f"Resuming training from {resume}")
        model = YOLO(resume)
    else:
        weights = f'{model_size}.pt' if pretrained else f'{model_size}.yaml'
        logger.info(f"Starting training: {weights}")
        model = YOLO(weights)

    # Log hardware info
    import torch
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name} ({props.total_memory / 1e9:.1f}GB)")

    # Validate dataset
    if not validate_dataset(Path(data_yaml)):
        logger.warning("Dataset validation failed. Training may fail.")
        logger.warning("Prepare dataset first: python scripts/prepare_doclaynet.py")

    # Train
    logger.info(f"Training {model_size} for {epochs} epochs")
    logger.info(f"Batch size: {batch}, Image size: {imgsz}")
    t0 = time.time()

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        pretrained=pretrained,
        patience=patience,
        save_period=save_period,
        workers=workers,
        verbose=True,
        # Augmentation (matching handwriting_detection.yaml)
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.1,
        # Optimization
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed/3600:.1f}h")

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_results = model.val(data=data_yaml, device=device)

    best_weights = Path(project) / name / 'weights' / 'best.pt'
    results_dict = {
        'model': model_size,
        'epochs': epochs,
        'training_time_hours': elapsed / 3600,
        'best_weights': str(best_weights),
        'val_metrics': {
            'mAP50': float(val_results.box.map50) if hasattr(val_results, 'box') else None,
            'mAP50_95': float(val_results.box.map) if hasattr(val_results, 'box') else None,
            'precision': float(val_results.box.mp) if hasattr(val_results, 'box') else None,
            'recall': float(val_results.box.mr) if hasattr(val_results, 'box') else None,
        },
    }

    # Save training summary
    summary_path = Path(project) / name / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Training summary saved -> {summary_path}")

    # Report results
    m = results_dict['val_metrics']
    logger.info(f"\n{'='*50}")
    logger.info(f"Training Results ({model_size}):")
    logger.info(f"  mAP@0.5:      {m.get('mAP50', 'N/A')}")
    logger.info(f"  mAP@0.5:0.95: {m.get('mAP50_95', 'N/A')}")
    logger.info(f"  Precision:    {m.get('precision', 'N/A')}")
    logger.info(f"  Recall:       {m.get('recall', 'N/A')}")
    logger.info(f"  Best weights: {best_weights}")
    logger.info(f"{'='*50}\n")

    return results_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 handwriting detector")
    parser.add_argument('--data', type=str, default='configs/handwriting_detection.yaml')
    parser.add_argument('--model', type=str, default='yolov8x',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--project', type=str, default='runs/detect')
    parser.add_argument('--name', type=str, default='baseline_v1')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--save-period', type=int, default=10)
    parser.add_argument('--workers', type=int, default=8)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_yolov8(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=not args.no_pretrained,
        resume=args.resume,
        patience=args.patience,
        save_period=args.save_period,
        workers=args.workers,
    )
