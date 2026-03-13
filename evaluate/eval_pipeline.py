"""
End-to-end pipeline evaluation on lecture slide images.

Evaluates the full system (Detection → OCR → Replacement) on annotated slides.

Metrics:
    - Detection: mAP@0.5, precision, recall per class (text/math)
    - Text OCR: CER, WER (with/without German post-processing)
    - Math OCR: BLEU
    - End-to-end: Combined score

Usage:
    # Evaluate baseline pipeline
    python evaluate/eval_pipeline.py \
        --detector runs/detect/baseline_v1/weights/best.pt \
        --ocr microsoft/trocr-large-handwritten \
        --data data/processed/detection/dataset.yaml \
        --german-test data/processed/german_text/german_text_test.json \
        --output outputs/eval_pipeline_baseline.json

    # Evaluate Phase 2 pipeline (DLAFormer + fine-tuned TrOCR + TAMER)
    python evaluate/eval_pipeline.py \
        --detector dlaformer \
        --ocr checkpoints/trocr_german/best \
        --math tamer \
        --output outputs/eval_pipeline_phase2.json

    # With professor adaptation (Phase 3)
    python evaluate/eval_pipeline.py \
        --detector dlaformer \
        --ocr checkpoints/trocr_german/best \
        --meta-checkpoint checkpoints/meta_learning/meta_checkpoint_best.pt \
        --adaptation-samples path/to/professor_samples.json \
        --output outputs/eval_pipeline_adapted.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger

from utils.metrics import batch_cer, batch_wer, compute_bleu
from utils.image_utils import load_image


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------

def build_pipeline(
    detector_type: str,
    detector_path: Optional[str],
    ocr_model: str,
    math_model: str = 'pix2tex',
    meta_checkpoint: Optional[str] = None,
    device: str = 'cuda',
):
    """
    Build the full OCR pipeline.

    Args:
        detector_type: 'yolov8' or 'dlaformer'
        detector_path: Path to detector checkpoint (for YOLOv8)
        ocr_model: TrOCR model ID or checkpoint path
        math_model: 'pix2tex' or 'tamer'
        meta_checkpoint: Path to MAML meta-checkpoint (optional)
        device: Compute device

    Returns:
        Pipeline instance (BaselinePipeline or adapted variant)
    """
    from baseline.baseline_pipeline import (
        YOLOv8Detector, TrOCRRecognizer, Pix2TexRecognizer, BaselinePipeline
    )

    # Detector
    if detector_type == 'dlaformer':
        try:
            from models.dlaformer_adapter import DLAFormerDetector
            detector = DLAFormerDetector(device=device)
            logger.info("Using DLAFormer detector")
        except Exception as e:
            logger.warning(f"DLAFormer failed ({e}), falling back to YOLOv8")
            detector = YOLOv8Detector(model_path=detector_path or 'yolov8x.pt')
    else:
        detector = YOLOv8Detector(model_path=detector_path or 'yolov8x.pt')
        logger.info(f"Using YOLOv8 detector: {detector_path}")

    # Text OCR
    if meta_checkpoint and Path(meta_checkpoint).exists():
        from models.meta_learning_ocr import MAMLOCRWrapper
        import torch
        wrapper = MAMLOCRWrapper(base_model_path=ocr_model, device=device)
        ckpt = torch.load(meta_checkpoint, map_location=device)
        wrapper.meta_model.load_state_dict(ckpt['meta_model_state'])
        logger.info(f"Using MAML meta-learned OCR, checkpoint: {meta_checkpoint}")
        text_recognizer = wrapper
    else:
        text_recognizer = TrOCRRecognizer(model_id=ocr_model, device=device)
        logger.info(f"Using TrOCR: {ocr_model}")

    # Math OCR
    if math_model == 'tamer':
        try:
            from models.math_ocr_tamer import TAMERMathOCR
            math_recognizer = TAMERMathOCR(device=device)
            logger.info(f"Using TAMER math OCR (type: {math_recognizer.model_type})")
        except Exception as e:
            logger.warning(f"TAMER failed ({e}), falling back to Pix2Tex")
            math_recognizer = Pix2TexRecognizer()
    else:
        math_recognizer = Pix2TexRecognizer()
        logger.info("Using Pix2Tex math OCR")

    pipeline = BaselinePipeline(
        detector=detector,
        text_recognizer=text_recognizer,
        math_recognizer=math_recognizer,
    )
    return pipeline


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_detection_subset(
    pipeline,
    data_yaml: str,
    split: str = 'test',
    conf: float = 0.25,
    iou: float = 0.45,
    output_path: Optional[Path] = None,
) -> dict:
    """Run detection-only evaluation using YOLOv8 val."""
    from evaluate.eval_detection import evaluate_detector

    detector = pipeline.detector
    if not hasattr(detector, 'model_path'):
        logger.warning("Detection eval requires YOLOv8 detector with model_path attribute")
        return {}

    return evaluate_detector(
        model_path=detector.model_path,
        data_yaml=data_yaml,
        split=split,
        conf=conf,
        iou=iou,
        output_path=output_path,
    )


def evaluate_german_ocr_subset(
    pipeline,
    test_manifest: Path,
    batch_size: int = 8,
    postprocess: bool = True,
    max_samples: int = 200,
    adaptation_samples: Optional[List[Dict]] = None,
) -> dict:
    """
    Evaluate German text OCR on manifest.

    Optionally adapts MAML model to professor samples first.
    """
    with open(test_manifest, encoding='utf-8') as f:
        data = json.load(f)
    samples = data.get('samples', [])[:max_samples]
    logger.info(f"Evaluating German OCR on {len(samples)} samples")

    # Adapt if meta-learning model and professor samples provided
    if adaptation_samples and hasattr(pipeline.text_recognizer, 'adapt'):
        logger.info(f"Adapting to professor: {len(adaptation_samples)} samples")
        pipeline.text_recognizer.adapt(adaptation_samples)

    hypotheses = []
    references = []

    for sample in samples:
        img_path = Path(sample['image'])
        if not img_path.is_absolute():
            img_path = Path('data') / img_path
        if not img_path.exists():
            continue

        try:
            img = load_image(img_path, mode='rgb')
            if hasattr(pipeline.text_recognizer, 'predict'):
                # MAML wrapper
                preds = pipeline.text_recognizer.predict([img], postprocess_german=postprocess)
                hyp = preds[0] if preds else ''
            else:
                # Standard TrOCR
                from PIL import Image as PILImage
                pil = PILImage.fromarray(img)
                hyp = pipeline.text_recognizer.recognize(pil) or ''
                if postprocess:
                    from utils.german_postprocessing import correct_text
                    hyp = correct_text(hyp)
            hypotheses.append(hyp)
            references.append(sample['text'])
        except Exception as e:
            logger.debug(f"Error on {img_path}: {e}")

    if not hypotheses:
        return {}

    cer_metrics = batch_cer(hypotheses, references)
    wer_metrics = batch_wer(hypotheses, references)

    return {
        'n_samples': len(hypotheses),
        'CER': cer_metrics['mean_cer'],
        'WER': wer_metrics['mean_wer'],
        'CER_percent': cer_metrics['mean_cer'] * 100,
        'WER_percent': wer_metrics['mean_wer'] * 100,
        'examples': [
            {'hypothesis': h, 'reference': r}
            for h, r in zip(hypotheses[:10], references[:10])
        ],
    }


def evaluate_math_ocr_subset(
    pipeline,
    test_manifest: Path,
    max_samples: int = 100,
) -> dict:
    """Evaluate math OCR on manifest."""
    with open(test_manifest, encoding='utf-8') as f:
        data = json.load(f)
    samples = data.get('samples', [])[:max_samples]

    if not samples:
        logger.warning(f"No math test samples in {test_manifest}")
        return {}

    hypotheses = []
    references = []

    for sample in samples:
        img_path = Path(sample.get('image', ''))
        if not img_path.exists():
            img_path = Path('data') / img_path
        if not img_path.exists():
            continue
        try:
            img = load_image(img_path, mode='rgb')
            latex = pipeline.math_recognizer.recognize(img)
            hypotheses.append(latex)
            references.append(sample.get('latex', ''))
        except Exception as e:
            logger.debug(f"Math OCR error: {e}")

    if not hypotheses:
        return {}

    bleu_scores = [compute_bleu(h, r) for h, r in zip(hypotheses, references)]
    return {
        'n_samples': len(hypotheses),
        'BLEU': float(np.mean(bleu_scores)),
        'BLEU_percent': float(np.mean(bleu_scores)) * 100,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_full_pipeline(
    detector_type: str = 'yolov8',
    detector_path: Optional[str] = None,
    ocr_model: str = 'microsoft/trocr-large-handwritten',
    math_model: str = 'pix2tex',
    data_yaml: str = 'configs/handwriting_detection.yaml',
    german_test: Path = Path('data/processed/german_text/german_text_test.json'),
    math_test: Path = Path('data/processed/math/math_test.json'),
    meta_checkpoint: Optional[str] = None,
    adaptation_samples_path: Optional[str] = None,
    device: str = 'cuda',
    output_path: Optional[Path] = None,
) -> dict:
    """
    Full pipeline evaluation.

    Returns combined metrics dict covering detection, German OCR, and math OCR.
    """
    logger.info("=" * 60)
    logger.info("Full Pipeline Evaluation")
    logger.info(f"  Detector:     {detector_type} ({detector_path or 'default'})")
    logger.info(f"  Text OCR:     {ocr_model}")
    logger.info(f"  Math OCR:     {math_model}")
    logger.info(f"  Meta-ckpt:    {meta_checkpoint or 'None'}")
    logger.info("=" * 60)

    # Load adaptation samples if provided
    adaptation_samples = None
    if adaptation_samples_path and Path(adaptation_samples_path).exists():
        with open(adaptation_samples_path) as f:
            adaptation_samples = json.load(f).get('samples', [])
        logger.info(f"Loaded {len(adaptation_samples)} professor adaptation samples")

    # Build pipeline
    t0 = time.time()
    pipeline = build_pipeline(
        detector_type=detector_type,
        detector_path=detector_path,
        ocr_model=ocr_model,
        math_model=math_model,
        meta_checkpoint=meta_checkpoint,
        device=device,
    )
    logger.info(f"Pipeline built in {time.time()-t0:.1f}s")

    all_metrics = {
        'config': {
            'detector': detector_type,
            'detector_path': str(detector_path) if detector_path else None,
            'ocr_model': ocr_model,
            'math_model': math_model,
            'meta_checkpoint': meta_checkpoint,
        }
    }

    # Detection evaluation
    if detector_path and Path(detector_path).exists():
        logger.info("\n--- Detection Evaluation ---")
        det_metrics = evaluate_detection_subset(
            pipeline=pipeline,
            data_yaml=data_yaml,
            output_path=output_path.parent / 'detection_results.json' if output_path else None,
        )
        all_metrics['detection'] = det_metrics
        if det_metrics:
            logger.info(f"  mAP@0.5:  {det_metrics.get('mAP50', 'N/A'):.4f}")
            logger.info(f"  Precision: {det_metrics.get('precision', 'N/A'):.4f}")
            logger.info(f"  Recall:    {det_metrics.get('recall', 'N/A'):.4f}")

    # German OCR evaluation
    if german_test.exists():
        logger.info("\n--- German OCR Evaluation ---")
        ocr_metrics = evaluate_german_ocr_subset(
            pipeline=pipeline,
            test_manifest=german_test,
            postprocess=True,
            max_samples=500,
            adaptation_samples=adaptation_samples,
        )
        all_metrics['german_ocr'] = ocr_metrics
        if ocr_metrics:
            logger.info(f"  CER: {ocr_metrics.get('CER_percent', 'N/A'):.2f}%")
            logger.info(f"  WER: {ocr_metrics.get('WER_percent', 'N/A'):.2f}%")

    # Math OCR evaluation
    if math_test.exists():
        logger.info("\n--- Math OCR Evaluation ---")
        math_metrics = evaluate_math_ocr_subset(
            pipeline=pipeline,
            test_manifest=math_test,
        )
        all_metrics['math_ocr'] = math_metrics
        if math_metrics:
            logger.info(f"  BLEU: {math_metrics.get('BLEU_percent', 'N/A'):.1f}%")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Evaluation Summary")
    if 'detection' in all_metrics and all_metrics['detection']:
        d = all_metrics['detection']
        logger.info(f"  Detection mAP@0.5:  {d.get('mAP50', 'N/A'):.4f}")
    if 'german_ocr' in all_metrics and all_metrics['german_ocr']:
        g = all_metrics['german_ocr']
        logger.info(f"  German CER: {g.get('CER_percent', 'N/A'):.2f}%")
        logger.info(f"  German WER: {g.get('WER_percent', 'N/A'):.2f}%")
    if 'math_ocr' in all_metrics and all_metrics['math_ocr']:
        m = all_metrics['math_ocr']
        logger.info(f"  Math BLEU:  {m.get('BLEU_percent', 'N/A'):.1f}%")
    logger.info("=" * 60)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved -> {output_path}")

    return all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Full pipeline evaluation")
    parser.add_argument('--detector', type=str, default='yolov8',
                        choices=['yolov8', 'dlaformer'],
                        help='Detector type')
    parser.add_argument('--detector-path', type=str, default=None,
                        help='Path to detector checkpoint (required for yolov8)')
    parser.add_argument('--ocr', type=str,
                        default='microsoft/trocr-large-handwritten',
                        help='TrOCR model ID or checkpoint path')
    parser.add_argument('--math', type=str, default='pix2tex',
                        choices=['pix2tex', 'tamer'],
                        help='Math OCR model')
    parser.add_argument('--data', type=str,
                        default='configs/handwriting_detection.yaml',
                        help='YOLO dataset yaml for detection eval')
    parser.add_argument('--german-test', type=Path,
                        default=Path('data/processed/german_text/german_text_test.json'))
    parser.add_argument('--math-test', type=Path,
                        default=Path('data/processed/math/math_test.json'))
    parser.add_argument('--meta-checkpoint', type=str, default=None,
                        help='Path to MAML meta-checkpoint for professor adaptation')
    parser.add_argument('--adaptation-samples', type=str, default=None,
                        help='Path to JSON with professor support samples')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/eval_pipeline.json'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_full_pipeline(
        detector_type=args.detector,
        detector_path=args.detector_path,
        ocr_model=args.ocr,
        math_model=args.math,
        data_yaml=args.data,
        german_test=args.german_test,
        math_test=args.math_test,
        meta_checkpoint=args.meta_checkpoint,
        adaptation_samples_path=args.adaptation_samples,
        device=args.device,
        output_path=args.output,
    )
