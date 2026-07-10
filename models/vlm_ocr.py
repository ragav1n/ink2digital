"""
Qwen3-VL OCR backend for the lecture-slide pipeline.

Drop-in replacement for `MAMLOCRWrapper.predict` (see
`models/meta_learning_ocr.py:325-359`). When wired into `infer.py` with
`--ocr-backend vlm`, each detected crop is sent to Qwen3-VL-8B with a
tight transcription prompt. The staged pipeline (detection, typeset
filter, ink-erase render, cross-source NMS) is unchanged — only the
recognizer is swapped.

Design choices:
- Lazy load on first `.predict()` call to keep CLI startup fast.
- Per-crop inference (no batching) — batching multiple variable-sized
  crops inflates VRAM unpredictably on the 16 GB V100.
- 8-bit via bitsandbytes by default (~12 GB) to coexist with the
  detector/optional LLM corrector loaders.
- `max_pixels` + `resize_max_edge` cap the vision-token count. Without
  these, a 3840x2160 lecture slide produced ~35K vision tokens and OOMed
  the attention matrix at 73 GiB on the V100 (established in
  `scripts/vlm_baseline.py` on 2026-05-13).
- The `use_adapted` and `postprocess_german` kwargs are accepted for
  signature parity with MAMLOCRWrapper, but ignored — VLM output is
  already clean German / LaTeX, and there is no Reptile adapter for the
  VLM in Phase 1.
"""

from __future__ import annotations

import gc
from typing import List, Optional

import numpy as np
from loguru import logger


_TEXT_PROMPT = (
    "Transcribe the handwritten German text in this image. "
    "Output ONLY the transcription, no explanation, no markdown, no quotes. "
    "If the line is mathematical notation, output it as LaTeX without $ delimiters."
)

_MATH_PROMPT = (
    "Transcribe the handwritten mathematics in this image as LaTeX. "
    "Wrap any natural-language words (e.g. German text or labels) in \\text{...} "
    "so spacing is preserved. "
    "If there are multiple equation lines, separate them with a LaTeX row "
    "break \\\\, not a newline character. "
    "Output ONLY the LaTeX code, no $ delimiters, no explanation, no markdown."
)


class VLMOCRBackend:
    """Qwen3-VL-based OCR backend exposing the MAMLOCRWrapper.predict shape."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        load_in_8bit: bool = True,
        max_pixels: int = 1024 * 1024,
        resize_max_edge: int = 1280,
        max_new_tokens: int = 256,
        adapter_path: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.max_pixels = max_pixels
        self.resize_max_edge = resize_max_edge
        self.max_new_tokens = max_new_tokens
        # Optional per-professor LoRA adapter (Phase 2). When set, the adapter
        # is layered on top of the frozen base at load time.
        self.adapter_path = adapter_path
        self.processor = None
        self.model = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        logger.info(f"Loading VLM OCR backend: {self.model_id} "
                    f"(8bit={self.load_in_8bit}, max_pixels={self.max_pixels})")

        kwargs = {}
        if self.load_in_8bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = self.device
        else:
            kwargs["dtype"] = torch.bfloat16

        self.processor = AutoProcessor.from_pretrained(self.model_id, max_pixels=self.max_pixels)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
        if not self.load_in_8bit:
            self.model = self.model.to(self.device)
        if self.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            logger.info(f"Loaded per-professor LoRA adapter: {self.adapter_path}")
        self.model.eval()
        logger.info(f"VLM OCR backend ready on {self.device}")

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
        logger.info("VLM OCR backend unloaded; GPU memory freed.")

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
        gen = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.processor.decode(gen, skip_special_tokens=True).strip()
        # Strip wrapping quotes / code fences if the model added any despite the prompt.
        for fence in ("```latex", "```math", "```"):
            if text.startswith(fence):
                text = text[len(fence):].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ("\"", "'"):
            text = text[1:-1].strip()
        return text

    def predict(
        self,
        images: List[np.ndarray],
        use_adapted: bool = False,
        postprocess_german: bool = False,
        mode: str = "text",
    ) -> List[str]:
        """Match MAMLOCRWrapper.predict signature; `use_adapted` and
        `postprocess_german` are ignored (kept for parity).

        Extra arg `mode` selects the prompt: 'text' for German handwriting,
        'math' for LaTeX. Default 'text'.
        """
        del use_adapted, postprocess_german  # unused — signature parity only
        if not images:
            return []
        self._ensure_loaded()
        prompt = _MATH_PROMPT if mode == "math" else _TEXT_PROMPT
        out = []
        for crop in images:
            try:
                out.append(self._run_one(crop, prompt))
            except Exception as e:
                logger.warning(f"  VLM OCR failed on crop {crop.shape}: {e}")
                out.append("")
        return out
