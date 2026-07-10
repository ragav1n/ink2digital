"""DocLayout-YOLO fine-tune for the Jakob handwritten-region detector.

Added for the IEEE Access resubmission (Reviewer 1, comment 3): a
document-specialised detection baseline alongside YOLOv8x/YOLO26x/RT-DETR on
the v2 Jakob dataset (165 slides; 132 train / 33 val, text=0, math=1).

DocLayout-YOLO (Zhao et al., 2024) is a YOLOv10 fork pre-trained on
DocStructBench, i.e. a document-domain warm start rather than a COCO one —
which makes it the most interesting of the reviewer-requested baselines given
the paper's checkpoint-continuity argument.

Requires its own package (a fork, coexists with ultralytics):
    pip install doclayout-yolo huggingface_hub

Same protocol as ``finetune_jakob_detector_yolo26.py`` (imgsz=1536,
document-friendly augmentation, same optimisation settings):

    YOLOv8x  hires   -> runs/jakob_detector_v3/hires1536/       (mAP50 ~= 0.54)
    YOLO26x  hires   -> runs/jakob_detector_v3/yolo26x_v1/      (mAP50 ~= 0.37)
    DocLayout-YOLO   -> runs/jakob_detector_v3/doclayout_v1/    (this script)

Usage (on the RTX 4060 Ti desktop):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        python training/finetune_jakob_detector_doclayout.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger


_HF_REPO = 'juliozhao/DocLayout-YOLO-DocStructBench'
_HF_FILE = 'doclayout_yolo_docstructbench_imgsz1024.pt'


def _fetch_weights() -> str:
    from huggingface_hub import hf_hub_download
    logger.info(f"Fetching DocStructBench weights: {_HF_REPO}/{_HF_FILE}")
    return hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='configs/jakob_detection_v2.yaml')
    p.add_argument('--model', type=str, default=None,
                   help='Path to DocLayout-YOLO weights. Default: download the '
                        'DocStructBench checkpoint from the HF hub.')
    p.add_argument('--imgsz', type=int, default=1536)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--project', type=str, default='runs/jakob_detector_v3')
    p.add_argument('--name', type=str, default='doclayout_v1')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    from doclayout_yolo import YOLOv10

    weights = args.model or _fetch_weights()
    logger.info(f"DocLayout-YOLO Jakob detector fine-tune: imgsz={args.imgsz}, "
                f"batch={args.batch}, from {weights}")
    model = YOLOv10(weights)

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
        # sheared, flipped or mixed. Matches the v8x/YOLO26x baselines.
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
        'model': str(weights),
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
