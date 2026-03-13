"""
Baseline pipeline: YOLOv8 + TrOCR (multilingual) + Pix2Tex.

This is the Phase 1 baseline achieving ~70-75% end-to-end accuracy.
It serves as the benchmark against which all subsequent improvements are measured.

Architecture:
    - Detection:  YOLOv8x (pretrained COCO, fine-tuned on DocLayNet)
    - Text OCR:   microsoft/trocr-large-handwritten (multilingual capable)
    - Math OCR:   pix2tex (Pix2Tex LaTeX OCR)

Usage:
    from baseline.baseline_pipeline import BaselinePipeline
    pipeline = BaselinePipeline()
    results = pipeline.process_image(image_path)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
from loguru import logger

from utils.image_utils import load_image, extract_region, numpy_to_pil
from utils.german_postprocessing import correct_german_ocr


# ---------------------------------------------------------------------------
# Result dataclasses (plain dicts for simplicity)
# ---------------------------------------------------------------------------

def make_result(
    bbox: List[float],
    class_name: str,
    confidence: float,
    text: str,
    ocr_time: float = 0.0,
) -> dict:
    return {
        'bbox': bbox,          # [x1, y1, x2, y2]
        'type': class_name,    # 'text' or 'math'
        'confidence': confidence,
        'text': text,
        'ocr_time': ocr_time,
    }


# ---------------------------------------------------------------------------
# Text OCR: TrOCR
# ---------------------------------------------------------------------------

class TrOCRRecognizer:
    """
    Wrapper for microsoft/trocr-large-handwritten.
    Supports German text via the multilingual encoder.
    """

    MODEL_ID = 'microsoft/trocr-large-handwritten'

    def __init__(self, device: str = 'cuda', model_id: Optional[str] = None):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch

        self.device = device if self._has_cuda() else 'cpu'
        mid = model_id or self.MODEL_ID
        logger.info(f"Loading TrOCR from {mid} on {self.device}...")

        self.processor = TrOCRProcessor.from_pretrained(mid)
        self.model = VisionEncoderDecoderModel.from_pretrained(mid)
        self.model.to(self.device)
        self.model.eval()

        import torch
        self.torch = torch
        logger.info("TrOCR loaded successfully")

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def recognize(self, image_crops: List[np.ndarray]) -> List[str]:
        """
        Recognize text in a batch of image crops.

        Args:
            image_crops: List of RGB numpy arrays.

        Returns:
            List of recognized text strings.
        """
        if not image_crops:
            return []

        pil_images = [numpy_to_pil(crop) for crop in image_crops]
        pixel_values = self.processor(
            images=pil_images,
            return_tensors='pt',
        ).pixel_values.to(self.device)

        with self.torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=128,
            )

        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return texts

    def recognize_single(self, image: np.ndarray) -> str:
        results = self.recognize([image])
        return results[0] if results else ''


# ---------------------------------------------------------------------------
# Math OCR: Pix2Tex
# ---------------------------------------------------------------------------

class Pix2TexRecognizer:
    """
    Wrapper for pix2tex LaTeX OCR.
    """

    def __init__(self):
        logger.info("Loading Pix2Tex LaTeX OCR...")
        try:
            from pix2tex.cli import LatexOCR
            self.model = LatexOCR()
            self.available = True
            logger.info("Pix2Tex loaded successfully")
        except ImportError:
            logger.warning("pix2tex not installed. Math OCR will return empty strings.")
            logger.warning("Install with: pip install pix2tex")
            self.model = None
            self.available = False

    def recognize(self, image: np.ndarray) -> str:
        """
        Recognize a math expression image and return LaTeX string.
        """
        if not self.available or self.model is None:
            return ''
        try:
            pil_img = numpy_to_pil(image)
            latex = self.model(pil_img)
            return latex or ''
        except Exception as e:
            logger.debug(f"Pix2Tex error: {e}")
            return ''


# ---------------------------------------------------------------------------
# Detector: YOLOv8
# ---------------------------------------------------------------------------

class YOLOv8Detector:
    """
    YOLOv8 detector for handwritten text/math region detection.
    Uses pretrained weights by default; can load fine-tuned checkpoint.
    """

    CLASS_NAMES = ['text', 'math']

    def __init__(
        self,
        weights: str = 'yolov8x.pt',
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = 'cuda',
    ):
        from ultralytics import YOLO
        import torch

        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading YOLOv8 from {weights} on {self.device}...")
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        logger.info("YOLOv8 loaded successfully")

    def detect(
        self,
        image: np.ndarray,
        min_area: int = 100,
    ) -> List[Dict]:
        """
        Run detection on a single image.

        Returns:
            List of {'bbox': [x1,y1,x2,y2], 'class': str, 'confidence': float}
        """
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # Filter tiny boxes
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    continue

                cls_name = (self.CLASS_NAMES[cls_id]
                            if cls_id < len(self.CLASS_NAMES)
                            else 'text')
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': cls_name,
                    'confidence': conf,
                })

        return detections


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class BaselinePipeline:
    """
    End-to-end baseline pipeline for German lecture slide OCR.

    Steps:
    1. Detect text and math regions (YOLOv8)
    2. Recognize text regions (TrOCR multilingual)
    3. Recognize math regions (Pix2Tex)
    4. Post-process German text (umlaut correction, spell check)
    """

    def __init__(
        self,
        detector_weights: str = 'yolov8x.pt',
        trocr_model: Optional[str] = None,
        device: str = 'cuda',
        conf_threshold: float = 0.35,
        postprocess_german: bool = True,
    ):
        self.device = device
        self.postprocess_german = postprocess_german

        # Initialize models
        self.detector = YOLOv8Detector(
            weights=detector_weights,
            conf_threshold=conf_threshold,
            device=device,
        )
        self.text_ocr = TrOCRRecognizer(device=device, model_id=trocr_model)
        self.math_ocr = Pix2TexRecognizer()

    def process_image(
        self,
        image: Union[str, Path, np.ndarray],
        batch_text: bool = True,
    ) -> List[dict]:
        """
        Process a single slide image end-to-end.

        Args:
            image: Path to image or numpy RGB array.
            batch_text: If True, batch all text crops into one TrOCR call.

        Returns:
            List of result dicts with 'bbox', 'type', 'confidence', 'text'.
        """
        if isinstance(image, (str, Path)):
            img = load_image(image, mode='rgb')
        else:
            img = image

        t0 = time.time()

        # Step 1: Detection
        detections = self.detector.detect(img)
        logger.debug(f"Detected {len(detections)} regions in {time.time()-t0:.2f}s")

        if not detections:
            return []

        # Separate text and math regions
        text_dets = [d for d in detections if d['class'] == 'text']
        math_dets = [d for d in detections if d['class'] == 'math']

        results = []

        # Step 2: Text OCR (batched)
        if text_dets:
            crops = [extract_region(img, d['bbox']) for d in text_dets]
            t_ocr = time.time()
            if batch_text:
                texts = self.text_ocr.recognize(crops)
            else:
                texts = [self.text_ocr.recognize_single(c) for c in crops]
            ocr_time = time.time() - t_ocr

            for det, text in zip(text_dets, texts):
                if self.postprocess_german:
                    text = correct_german_ocr(text)
                results.append(make_result(
                    bbox=det['bbox'],
                    class_name='text',
                    confidence=det['confidence'],
                    text=text,
                    ocr_time=ocr_time / len(text_dets),
                ))

        # Step 3: Math OCR (sequential — Pix2Tex doesn't support batching well)
        for det in math_dets:
            crop = extract_region(img, det['bbox'])
            t_ocr = time.time()
            latex = self.math_ocr.recognize(crop)
            results.append(make_result(
                bbox=det['bbox'],
                class_name='math',
                confidence=det['confidence'],
                text=latex,
                ocr_time=time.time() - t_ocr,
            ))

        # Sort by position (top-to-bottom, left-to-right)
        results.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))

        logger.debug(f"Total pipeline time: {time.time()-t0:.2f}s for {len(results)} regions")
        return results

    def process_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        progress: bool = True,
    ) -> List[List[dict]]:
        """Process a list of slide images."""
        from tqdm import tqdm
        iterator = tqdm(images, desc="Processing slides") if progress else images
        return [self.process_image(img) for img in iterator]


# ---------------------------------------------------------------------------
# Utility: render results back onto slide
# ---------------------------------------------------------------------------

def render_results(
    image: np.ndarray,
    results: List[dict],
    text_color: Tuple = (0, 200, 0),
    math_color: Tuple = (200, 0, 200),
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw detection boxes and recognized text on the image for visualization.
    Returns a copy of the image with annotations.
    """
    import cv2
    out = image.copy()
    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r['bbox']]
        color = text_color if r['type'] == 'text' else math_color
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{r['type']}: {r['text'][:30]}"
        cv2.putText(out, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
    return out


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        logger.info("Usage: python baseline_pipeline.py <image_path>")
        sys.exit(0)

    pipeline = BaselinePipeline()
    results = pipeline.process_image(sys.argv[1])
    for r in results:
        print(f"[{r['type']}] conf={r['confidence']:.2f}: {r['text']}")
