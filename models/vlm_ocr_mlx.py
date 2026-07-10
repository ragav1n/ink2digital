"""MLX VLM OCR backend — Qwen3-VL on Apple Silicon.

Drop-in counterpart to `models.vlm_ocr.VLMOCRBackend` for Macs. The CUDA backend
loads Qwen3-VL via `transformers` + `bitsandbytes` 8-bit, which is CUDA-only;
this one runs the same model through Apple's **MLX** framework instead, using a
pre-quantized 8-bit MLX checkpoint (same quantization level as the server, so no
quantization-driven quality loss).

`infer.py:load_pipeline` selects this class automatically when the resolved
device is not CUDA. The detection / typeset-filter / rendering pipeline is
unchanged — only the recognizer is swapped.

Interface parity: `predict(images, use_adapted, postprocess_german, mode)` and
`unload()` match `VLMOCRBackend`; `_run_one` and `model_id` attributes exist so
`infer.run_infer` detects this as a VLM backend
(`hasattr(ocr, '_run_one') and hasattr(ocr, 'model_id')`).

The Jakob LoRA adapter cannot be layered at load time here — bake it into the
model file with `scripts/fuse_and_convert_vlm_mlx.py` and pass the result as
`model_id`.

NOTE (verify on first run): the `mlx-vlm` API has changed across releases —
`generate`'s return type (str vs result object), `apply_chat_template`'s
signature, and how images are passed. Pin `mlx-vlm` in `requirements-mac.txt`
and confirm the calls in `_run_one` / `_ensure_loaded` against that version.
"""
from __future__ import annotations

import gc
from typing import List, Optional

import numpy as np
from loguru import logger

# Reuse the exact prompts the CUDA backend uses — keep recognition behaviour
# identical across backends.
from models.vlm_ocr import _MATH_PROMPT, _TEXT_PROMPT


class MLXVLMOCRBackend:
    """Qwen3-VL OCR backend running on MLX (Apple Silicon)."""

    def __init__(
        self,
        model_id: str = "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit",
        device: str = "mps",
        resize_max_edge: int = 1280,
        max_new_tokens: int = 256,
        **_ignored,  # load_in_8bit / max_pixels / adapter_path — parity only
    ):
        self.model_id = model_id
        self.device = device  # informational; MLX always uses the Apple GPU
        self.resize_max_edge = resize_max_edge
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self.config = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        logger.info(f"Loading MLX VLM OCR backend: {self.model_id}")
        self.model, self.processor = load(self.model_id)
        self.config = load_config(self.model_id)
        logger.info("MLX VLM OCR backend ready")

    def unload(self) -> None:
        if self.model is None:
            return
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self.config = None
        gc.collect()
        logger.info("MLX VLM OCR backend unloaded.")

    @staticmethod
    def _clean(text: str) -> str:
        """Strip code fences / wrapping quotes the model may add despite the prompt.

        Mirrors the post-processing in `VLMOCRBackend._run_one`.
        """
        text = (text or "").strip()
        for fence in ("```latex", "```math", "```"):
            if text.startswith(fence):
                text = text[len(fence):].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ("\"", "'"):
            text = text[1:-1].strip()
        return text

    def _run_one(self, crop: np.ndarray, prompt: str) -> str:
        import os
        import tempfile
        from PIL import Image as PILImage
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return ""

        # Downscale large slides — without this a 3840x2160 crop explodes the
        # vision-token count. Same cap as VLMOCRBackend.
        pil = PILImage.fromarray(crop)
        if self.resize_max_edge and max(pil.size) > self.resize_max_edge:
            pil.thumbnail((self.resize_max_edge, self.resize_max_edge), PILImage.LANCZOS)

        # mlx-vlm 0.5.0 `generate` takes image *paths* (Union[str, List[str]]),
        # not PIL objects — write the crop to a temp PNG.
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            pil.save(tmp_path)
            formatted = apply_chat_template(
                self.processor, self.config, prompt, num_images=1)
            result = generate(
                self.model, self.processor, formatted, [tmp_path],
                max_tokens=self.max_new_tokens, temperature=0.0, verbose=False,
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        # `generate` returns a GenerationResult; `.text` holds the output.
        # Fall back to str() for any other return shape.
        text = getattr(result, "text", None)
        if text is None:
            text = result if isinstance(result, str) else str(result)
        return self._clean(text)

    def predict(
        self,
        images: List[np.ndarray],
        use_adapted: bool = False,
        postprocess_german: bool = False,
        mode: str = "text",
    ) -> List[str]:
        """Match `VLMOCRBackend.predict`; `use_adapted`/`postprocess_german`
        are accepted for parity and ignored. `mode` selects the prompt."""
        del use_adapted, postprocess_german  # signature parity only
        if not images:
            return []
        self._ensure_loaded()
        prompt = _MATH_PROMPT if mode == "math" else _TEXT_PROMPT
        out: List[str] = []
        for crop in images:
            try:
                out.append(self._run_one(crop, prompt))
            except Exception as e:  # noqa: BLE001 — one bad crop must not abort the slide
                logger.warning(f"  MLX VLM OCR failed on crop {getattr(crop, 'shape', '?')}: {e}")
                out.append("")
        return out
