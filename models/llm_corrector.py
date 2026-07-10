"""
Local LLM corrector for German OCR output.

Loads Qwen3-8B-Instruct (or another HF causal-LM) and applies per-slide
spelling/grammar correction on German text fragments coming out of the
TrOCR/Jakob-FT pipeline. Math fragments are NEVER passed to this corrector
(the caller filters them out) — the LLM only ever sees German text lines.

Design choices:
- Lazy load: the model is ~16 GB in bf16, so we instantiate the class up
  front but defer the actual `from_pretrained` until `correct_slide` is
  first called. `unload()` frees the GPU.
- One prompt per slide: all text fragments for the slide go into a single
  JSON list; the model returns a corrected JSON list. This gives the model
  cross-fragment context (e.g. it can tell the slide is about Eigenwerte).
- Edit-distance guard: if the LLM's correction differs from the input by
  more than `max_edit_ratio` (default 30% of characters), we reject the
  correction and keep the original. Prevents hallucination from silently
  rewriting Jakob's actual phrasing.

Usage:
    corrector = LLMCorrector()
    corrected = corrector.correct_slide([
        {'idx': 0, 'text': 'Ableitungen 2. Ordsung.'},
        {'idx': 1, 'text': 'Tangentialiebene:'},
    ])
    corrector.unload()
"""

from __future__ import annotations

import gc
import json
import re
from typing import Dict, List, Optional

import editdistance
from loguru import logger


_SYSTEM_PROMPT = (
    "Du bist ein OCR-Korrekturassistent für deutsche Mathematik-Vorlesungsnotizen "
    "von Prof. Dr. Judith Jakob (FH Dortmund). Du erhältst OCR-Textfragmente "
    "als JSON-Liste. Korrigiere offensichtliche OCR-Fehler in deutschem Text.\n\n"
    "TYPISCHE OCR-FEHLER, DIE DU IMMER KORRIGIEREN SOLLST:\n"
    "  - Umlaute: 'fir' → 'für', 'fur' → 'für', 'uber' → 'über', 'mussen' → 'müssen'\n"
    "  - 'ss' → 'ß': 'heifst' → 'heißt', 'grosser' → 'größer', 'Strasse' → 'Straße'\n"
    "  - Vertauschte Buchstaben: 'Ordsung' → 'Ordnung', 'Bedingsungi' → 'Bedingung',\n"
    "    'Tangentialiebene' → 'Tangentialebene', 'indefiniteit' → 'indefinit',\n"
    "    'Unreichende' → 'Hinreichende', 'Unreichente' → 'Hinreichende'\n"
    "  - Großschreibung deutscher Substantive: 'mathematik' → 'Mathematik',\n"
    "    'informatik' → 'Informatik', 'matrix' → 'Matrix', 'eigenwerte' → 'Eigenwerte'\n"
    "  - Einzelne fehlende Buchstaben am Anfang: 'eine ...' bleibt, aber Fragmente\n"
    "    wie 'soiyo' bleiben unverändert (zu unsicher).\n\n"
    "WAS DU NICHT ÄNDERN DARFST:\n"
    "  - Fragmente, die Mathematik enthalten (z.B. 'fx = 2x', 'x^2 + y^2', '(0,0)')\n"
    "    bleiben EXAKT unverändert.\n"
    "  - Fragmente, die du nicht eindeutig als deutsches Wort erkennst (z.B. 'Sx-2x-xtzt.2',\n"
    "    'soiyo', 'xciyo'), bleiben unverändert.\n"
    "  - Eigennamen wie 'Jakob', 'FB Informatik', 'Modul 4,1061-' bleiben unverändert\n"
    "    (außer Tippfehler wie 'informatik' → 'Informatik').\n\n"
    "AUSGABEFORMAT: Antworte AUSSCHLIESSLICH mit einem JSON-Array der Form "
    '[{"idx": <int>, "text": "<korrigiert>"}], in derselben Reihenfolge und mit denselben '
    "idx-Werten wie die Eingabe. Keine Erklärung, kein Markdown, kein Kommentar — nur das JSON."
)


class LLMCorrector:
    """Wraps a HuggingFace causal-LM for German OCR-text correction."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-8B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        load_in_8bit: bool = False,
        max_edit_ratio: float = 0.30,
        max_new_tokens: int = 2048,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype_str = dtype
        self.load_in_8bit = load_in_8bit
        self.max_edit_ratio = max_edit_ratio
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading LLM: {self.model_id} (8bit={self.load_in_8bit})")

        kwargs: Dict = {}
        if self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                kwargs["device_map"] = self.device
            except ImportError as e:
                raise RuntimeError(
                    "load_in_8bit=True requires bitsandbytes. "
                    "pip install bitsandbytes"
                ) from e
        else:
            kwargs["dtype"] = getattr(torch, self.dtype_str)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if not self.load_in_8bit:
            self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"LLM loaded onto {self.device}")

    def unload(self) -> None:
        if self.model is None:
            return
        import torch
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("LLM unloaded; GPU memory freed.")

    @staticmethod
    def _edit_ratio(a: str, b: str) -> float:
        if not a:
            return 1.0 if b else 0.0
        return editdistance.eval(a, b) / max(len(a), 1)

    def correct_slide(self, fragments: List[Dict]) -> List[Dict]:
        """Correct a list of fragments. Each fragment must have 'idx' and 'text'.

        Returns a list of same length / same idx order. On any parse failure or
        upstream error, returns the input unchanged (safe fallback).
        """
        if not fragments:
            return fragments
        self._ensure_loaded()

        import torch

        user_payload = json.dumps(
            [{"idx": f["idx"], "text": f["text"]} for f in fragments],
            ensure_ascii=False,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ]

        # Qwen3 supports a <think>...</think> chain-of-thought mode by default;
        # disable it via the tokenizer flag to keep latency reasonable for this
        # short-form correction task. Falls back gracefully on non-Qwen models.
        chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **chat_kwargs
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, **chat_kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()

        parsed = self._parse_json_array(text)
        if parsed is None:
            logger.warning(f"  LLM output not parseable as JSON; keeping originals. "
                           f"Output head: {text[:200]!r}")
            return fragments

        # Build idx -> corrected text, then apply guard.
        by_idx = {item.get("idx"): item.get("text", "") for item in parsed if isinstance(item, dict)}
        out = []
        for f in fragments:
            orig = f["text"]
            new = by_idx.get(f["idx"], orig)
            if not isinstance(new, str):
                new = orig
            if new != orig:
                ratio = self._edit_ratio(orig, new)
                if ratio > self.max_edit_ratio:
                    logger.debug(f"    Rejected correction (edit_ratio={ratio:.2f} > "
                                 f"{self.max_edit_ratio}): {orig!r} -> {new!r}")
                    new = orig
            out.append({"idx": f["idx"], "text": new})
        return out

    @staticmethod
    def _parse_json_array(text: str) -> Optional[List]:
        """Extract the first JSON array from `text`. Tolerates ```json fences."""
        # Strip code fences if present
        fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        # Find the outermost [...] bracketed block
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
