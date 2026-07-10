"""PaddleOCR-VL-1.5 backend for math/text OCR.

Used as a baseline in ``evaluate/eval_math_ocr.py`` to position the Qwen3-VL
results against a second open VLM. PaddleOCR-VL-1.5 is a 0.9B
NaViT-encoder + ERNIE-4.5 decoder model with a built-in formula-recognition
prompt — small enough to load alongside other models on a 16 GB V100 without
quantization, and exposes the same HF ``AutoProcessor`` /
``AutoModelForImageTextToText`` interface as ``models/vlm_ocr.py``.

Source: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5

Design choices:
- Lazy load on first ``.predict()`` call.
- Per-crop inference; the model is tiny so batching is not a memory pressure
  but variable input sizes still complicate it — keep parity with VLMOCRBackend.
- BF16 by default. The model is too small to bother with 8-bit on the V100.
- Two prompts: ``"OCR:"`` for general text, ``"Formula Recognition:"`` for math.
  The math prompt is the documented entry point for LaTeX output.
"""

from __future__ import annotations

import gc
from typing import List, Optional

import numpy as np
from loguru import logger


# PaddleOCR-VL-1.5 task prompts (verified from HF model card).
_TEXT_PROMPT = "OCR:"
_MATH_PROMPT = "Formula Recognition:"


class PaddleOCRVLBackend:
    """PaddleOCR-VL-1.5 wrapper exposing the same ``predict`` shape as
    ``VLMOCRBackend``."""

    def __init__(
        self,
        model_id: str = "PaddlePaddle/PaddleOCR-VL-1.5",
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
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info(f"Loading PaddleOCR-VL backend: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
        ).to(self.device).eval()
        logger.info(f"PaddleOCR-VL backend ready on {self.device}")

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
        logger.info("PaddleOCR-VL backend unloaded.")

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
                    {"type": "image", "image": pil},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        # PaddleOCR-VL emits an EOS token at the end; the documented decode
        # slices it off with ``[:-1]``.
        gen = outputs[0][inputs["input_ids"].shape[-1]:-1]
        text = self.processor.decode(gen, skip_special_tokens=True).strip()
        # Strip wrapping code fences just in case.
        for fence in ("```latex", "```math", "```"):
            if text.startswith(fence):
                text = text[len(fence):].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        return text

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
                logger.warning(f"  PaddleOCR-VL failed on crop {crop.shape}: {e}")
                out.append("")
        return out
