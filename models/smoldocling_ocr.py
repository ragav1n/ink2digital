"""SmolDocling-256M backend for math/text OCR.

Added for the IEEE Access resubmission (Reviewer 1, comment 3): a lightweight
document VLM baseline for handwritten mathematical recognition, positioned
against Qwen3-VL-8B (8B) and PaddleOCR-VL-1.5 (0.9B) at the small end of the
parameter scale (256M).

SmolDocling (Nassar et al., ICCV 2025) is an ultra-compact document-conversion
VLM built on SmolVLM/Idefics3. It emits DocTags markup; the documented
formula-recognition entry point is the prompt ``"Convert formula to latex."``,
whose output we strip of DocTags/location tokens before scoring.

Source: https://huggingface.co/ds4sd/SmolDocling-256M-preview

Design choices (parity with ``models/paddleocr_vl_ocr.py``):
- Lazy load on first ``.predict()`` call.
- Per-crop inference; the model is tiny, batching buys nothing here.
- BF16 on CUDA, FP32 on CPU. Far too small to need quantisation.
"""

from __future__ import annotations

import gc
import re
from typing import List

import numpy as np
from loguru import logger


# Documented SmolDocling task prompts (HF model card).
_MATH_PROMPT = "Convert formula to latex."
_TEXT_PROMPT = "Convert this page to docling."

# DocTags cleanup: location tokens, tag wrappers, end-of-utterance marker.
_LOC_RE = re.compile(r"<loc_\d+>")
_TAG_RE = re.compile(r"</?[a-zA-Z_][^>]*>")


def _strip_doctags(s: str) -> str:
    """Reduce a DocTags string to its textual/LaTeX payload."""
    if not s:
        return ""
    s = s.replace("<end_of_utterance>", " ")
    s = _LOC_RE.sub(" ", s)
    s = _TAG_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


class SmolDoclingBackend:
    """SmolDocling-256M wrapper exposing the same ``predict`` shape as
    ``VLMOCRBackend``."""

    def __init__(
        self,
        model_id: str = "ds4sd/SmolDocling-256M-preview",
        device: str = "cuda",
        max_new_tokens: int = 512,
        resize_max_edge: int = 1280,
    ):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.resize_max_edge = resize_max_edge
        self.processor = None
        self.model = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        logger.info(f"Loading SmolDocling backend: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        ).to(self.device).eval()
        logger.info(f"SmolDocling backend ready on {self.device}")

    def unload(self) -> None:
        if self.model is None:
            return
        import torch
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SmolDocling backend unloaded.")

    def _run_one(self, crop: np.ndarray, prompt: str) -> str:
        import torch
        from PIL import Image as PILImage

        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return ""

        pil = PILImage.fromarray(crop)
        if self.resize_max_edge and max(pil.size) > self.resize_max_edge:
            pil.thumbnail((self.resize_max_edge, self.resize_max_edge), PILImage.LANCZOS)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=chat, images=[pil], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        gen = outputs[0][inputs["input_ids"].shape[-1]:]
        doctags = self.processor.decode(gen, skip_special_tokens=False)
        return _strip_doctags(doctags)

    def predict(
        self,
        images: List[np.ndarray],
        use_adapted: bool = False,
        postprocess_german: bool = False,
        mode: str = "text",
    ) -> List[str]:
        """Match ``VLMOCRBackend.predict`` signature; ``use_adapted`` and
        ``postprocess_german`` are accepted for parity and ignored."""
        del use_adapted, postprocess_german
        if not images:
            return []
        self._ensure_loaded()
        prompt = _MATH_PROMPT if mode == "math" else _TEXT_PROMPT
        out: List[str] = []
        for crop in images:
            try:
                out.append(self._run_one(crop, prompt))
            except Exception as e:
                logger.warning(f"  SmolDocling failed on crop {crop.shape}: {e}")
                out.append("")
        return out
