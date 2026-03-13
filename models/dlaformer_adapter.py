"""
DLAFormer adapter for handwritten text/math detection.
Phase 2 (Week 4-6): Upgrade from YOLOv8 baseline to transformer-based detection.

DLAFormer (Document Layout Analysis Transformer) from Microsoft achieves superior
performance on document layout detection tasks over CNN-based approaches.

This adapter wraps DLAFormer for our 2-class problem (text/math) and provides
a unified interface compatible with the baseline YOLOv8Detector API.

References:
    - DLAFormer: https://github.com/microsoft/DLAFormer
    - "Unifying Layout Analysis" (Microsoft, 2023)

Usage:
    from models.dlaformer_adapter import DLAFormerDetector
    detector = DLAFormerDetector(weights='checkpoints/dlaformer/best.pt')
    detections = detector.detect(image)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


# ---------------------------------------------------------------------------
# DLAFormer class mapping
# ---------------------------------------------------------------------------

# DocLayNet classes that DLAFormer was trained on -> our class ID
DLAFORMER_TO_OURS: Dict[str, Optional[int]] = {
    'Text': 0,
    'Caption': 0,
    'Footnote': 0,
    'List-item': 0,
    'Section-header': 0,
    'Title': 0,
    'Formula': 1,
    'Table': None,
    'Picture': None,
    'Page-header': None,
    'Page-footer': None,
}

OUR_CLASS_NAMES = ['text', 'math']


# ---------------------------------------------------------------------------
# DLAFormer wrapper
# ---------------------------------------------------------------------------

class DLAFormerDetector:
    """
    Adapter for the DLAFormer document layout analysis model.

    Wraps DLAFormer to provide a unified detection interface for
    text/math region detection on lecture slides.

    Args:
        weights: Path to DLAFormer checkpoint (.pt or .pth).
        config: Path to DLAFormer model config (YAML or JSON).
        device: CUDA device string or 'cpu'.
        conf_threshold: Minimum confidence for detection output.
        iou_threshold: IoU threshold for NMS.
        class_names: Override class names (default: DocLayNet).
    """

    def __init__(
        self,
        weights: Optional[str] = None,
        config: Optional[str] = None,
        device: str = 'cuda',
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.40,
        class_names: Optional[List[str]] = None,
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or list(DLAFORMER_TO_OURS.keys())

        self.model = self._load_model(weights, config)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, weights: Optional[str], config: Optional[str]) -> nn.Module:
        """
        Load DLAFormer model.

        Strategy:
        1. Try to load from local DLAFormer repo (external/DLAFormer)
        2. Fall back to HuggingFace if available
        3. Fall back to a simple DETR-based detection model
        """
        dlaformer_path = Path('external/DLAFormer')

        if dlaformer_path.exists():
            sys.path.insert(0, str(dlaformer_path.resolve()))
            try:
                return self._load_local_dlaformer(weights, config)
            except Exception as e:
                logger.warning(f"Local DLAFormer load failed: {e}")

        # Fallback: try HuggingFace
        try:
            return self._load_hf_dlaformer(weights)
        except Exception as e:
            logger.warning(f"HuggingFace DLAFormer load failed: {e}")

        # Fallback: use DETR fine-tuned on DocLayNet
        logger.warning("Falling back to facebook/detr-resnet-50 for layout detection")
        return self._load_detr_fallback(weights)

    def _load_local_dlaformer(self, weights: Optional[str], config: Optional[str]) -> nn.Module:
        """Load from local DLAFormer repository."""
        from dlaformer.model import DLAFormer as DLAFormerModel

        if config:
            model = DLAFormerModel.from_config(config)
        else:
            model = DLAFormerModel()

        if weights and Path(weights).exists():
            state = torch.load(weights, map_location=self.device)
            model.load_state_dict(state.get('model', state))
            logger.info(f"Loaded DLAFormer weights from {weights}")
        else:
            logger.info("Using DLAFormer with random initialization (needs fine-tuning)")

        return model

    def _load_hf_dlaformer(self, weights: Optional[str]) -> nn.Module:
        """Load DLAFormer from HuggingFace hub."""
        from transformers import AutoModelForObjectDetection
        model_id = weights or 'microsoft/dlaformer-large'
        logger.info(f"Loading DLAFormer from HuggingFace: {model_id}")
        model = AutoModelForObjectDetection.from_pretrained(model_id)
        return model

    def _load_detr_fallback(self, weights: Optional[str]) -> nn.Module:
        """Load DETR as fallback detection model."""
        from transformers import DetrForObjectDetection
        model_id = weights or 'facebook/detr-resnet-101'
        logger.info(f"Loading DETR fallback from: {model_id}")
        model = DetrForObjectDetection.from_pretrained(model_id)
        return model

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for DLAFormer input.
        Normalizes and reshapes to model input format.
        """
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        from utils.image_utils import numpy_to_pil, resize_image
        # Resize to model input size
        img = resize_image(image, width=1025, height=1025, keep_aspect=True)
        tensor = transform(numpy_to_pil(img))
        return tensor.unsqueeze(0).to(self.device)

    def postprocess(
        self,
        outputs,
        orig_h: int,
        orig_w: int,
    ) -> List[Dict]:
        """
        Post-process model outputs to detection dicts.

        Returns:
            List of {'bbox': [x1,y1,x2,y2], 'class': str, 'confidence': float}
        """
        detections = []

        # Handle DETR-style outputs
        if hasattr(outputs, 'logits') and hasattr(outputs, 'pred_boxes'):
            logits = outputs.logits[0]
            boxes = outputs.pred_boxes[0]

            probs = torch.softmax(logits, dim=-1)
            scores, labels = probs.max(dim=-1)

            for score, label, box in zip(scores, labels, boxes):
                score = float(score)
                label_idx = int(label)
                if score < self.conf_threshold:
                    continue

                # Get class name
                if label_idx < len(self.class_names):
                    cls_name = self.class_names[label_idx]
                else:
                    continue

                # Map to our class
                our_class = DLAFORMER_TO_OURS.get(cls_name)
                if our_class is None:
                    continue

                # Convert box from [cx, cy, w, h] normalized to [x1, y1, x2, y2]
                cx, cy, bw, bh = box.cpu().numpy()
                x1 = (cx - bw / 2) * orig_w
                y1 = (cy - bh / 2) * orig_h
                x2 = (cx + bw / 2) * orig_w
                y2 = (cy + bh / 2) * orig_h

                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class': OUR_CLASS_NAMES[our_class],
                    'confidence': score,
                })

        return self._apply_nms(detections)

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if len(detections) <= 1:
            return detections

        boxes = torch.tensor([d['bbox'] for d in detections], dtype=torch.float32)
        scores = torch.tensor([d['confidence'] for d in detections])

        from torchvision.ops import nms
        keep = nms(boxes, scores, self.iou_threshold)
        return [detections[i] for i in keep.tolist()]

    @torch.no_grad()
    def detect(
        self,
        image: np.ndarray,
        min_area: int = 100,
    ) -> List[Dict]:
        """
        Run DLAFormer detection on a single image.

        Args:
            image: RGB numpy array (H, W, 3).
            min_area: Minimum bounding box area in pixels.

        Returns:
            List of detection dicts.
        """
        h, w = image.shape[:2]
        pixel_values = self.preprocess(image)
        outputs = self.model(pixel_values=pixel_values)
        detections = self.postprocess(outputs, orig_h=h, orig_w=w)

        # Filter by minimum area
        detections = [
            d for d in detections
            if (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]) >= min_area
        ]
        return detections


# ---------------------------------------------------------------------------
# Fine-tuning wrapper
# ---------------------------------------------------------------------------

class DLAFormerTrainer:
    """
    Fine-tune DLAFormer on our handwriting detection dataset (DocLayNet + IAM).

    Uses the DETR-style training loop with:
    - Hungarian matching loss
    - Focal loss for classification
    - L1 + GIoU loss for boxes
    """

    def __init__(
        self,
        model: DLAFormerDetector,
        train_dataset,
        val_dataset,
        output_dir: str = 'checkpoints/dlaformer',
        learning_rate: float = 1e-4,
        num_epochs: int = 50,
        batch_size: int = 4,
        device: str = 'cuda',
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device

    def train(self):
        """Start fine-tuning. Full implementation in training/train_dlaformer.py."""
        raise NotImplementedError(
            "DLAFormer fine-tuning will be implemented in Phase 2 (Week 4). "
            "See training/train_dlaformer.py when ready."
        )


if __name__ == '__main__':
    # Quick load test
    logger.info("Testing DLAFormerDetector initialization...")
    try:
        detector = DLAFormerDetector(conf_threshold=0.4)
        test_img = np.ones((720, 1280, 3), dtype=np.uint8) * 200
        results = detector.detect(test_img)
        logger.info(f"Detection test: {len(results)} regions found on blank image")
    except Exception as e:
        logger.error(f"DLAFormer test failed: {e}")
        logger.info("DLAFormer requires Phase 2 setup. YOLOv8 baseline is fully functional.")
