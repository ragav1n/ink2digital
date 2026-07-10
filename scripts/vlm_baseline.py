"""
Zero-shot VLM baseline on held-out Jakob slides.

This is a RESEARCH BASELINE only. It is NOT wired into infer.py and does NOT
replace the staged pipeline (YOLOv8 + TrOCR/Reptile/Jakob-FT + TAMER + LLM
corrector). The point is to measure whether a general-purpose VLM matches
the specialized staged pipeline on the same held-out slides, so we have a
fair comparison number for the report/paper.

Model: Qwen/Qwen3-VL-8B-Instruct in 8-bit via bitsandbytes (same memory
budget as the LLM corrector we already use).

Usage:
    source venv/bin/activate
    python scripts/vlm_baseline.py
    # → outputs/jakob_vlm_baseline/page-XXXX.json + page-XXXX.raw.txt

Override the slide list:
    python scripts/vlm_baseline.py --images path1.jpg path2.jpg ...
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger


DEFAULT_HELD_OUT = [
    "data/Dr_Judith_Jakob_Slides/11 Extremwertberechnungen in mehreren Variablen_kommentiert images/"
    "11 Extremwertberechnungen in mehreren Variablen_kommentiert_page-0007.jpg",
    "data/Dr_Judith_Jakob_Slides/11 Extremwertberechnungen in mehreren Variablen_kommentiert images/"
    "11 Extremwertberechnungen in mehreren Variablen_kommentiert_page-0010.jpg",
    "data/Dr_Judith_Jakob_Slides/11 Extremwertberechnungen in mehreren Variablen_kommentiert images/"
    "11 Extremwertberechnungen in mehreren Variablen_kommentiert_page-0015.jpg",
    "data/Dr_Judith_Jakob_Slides/11 Extremwertberechnungen in mehreren Variablen_kommentiert images/"
    "11 Extremwertberechnungen in mehreren Variablen_kommentiert_page-0028.jpg",
]


PROMPT = (
    "Dies ist eine deutsche Mathematik-Vorlesungsfolie von Prof. Dr. Judith Jakob "
    "mit handschriftlichen Annotationen auf einer gedruckten Folie.\n\n"
    "Erkenne ALLE handschriftlichen Bereiche (Text und Mathematik) und gib für jeden "
    "ein JSON-Objekt zurück mit folgendem Schema:\n"
    "  - bbox: [x1, y1, x2, y2] (Pixelkoordinaten, ganzzahlig)\n"
    "  - type: 'text' (deutscher Fließtext) oder 'math' (mathematische Notation)\n"
    "  - content: bei 'text' der erkannte deutsche Text; bei 'math' der LaTeX-Code "
    "(ohne $-Zeichen)\n\n"
    "Beziehe nur HANDSCHRIFTLICHE Inhalte ein, nicht den bereits gedruckten Folientext.\n\n"
    "AUSGABEFORMAT: Antworte AUSSCHLIESSLICH mit einem JSON-Array, keine Erklärung, "
    "kein Markdown, kein Kommentar — nur das JSON-Array.\n"
    "Beispiel: [{\"bbox\":[120,340,580,420],\"type\":\"text\",\"content\":\"Hinreichendes Kriterium\"},"
    "{\"bbox\":[180,460,420,520],\"type\":\"math\",\"content\":\"f(x)=x^2+y^2\"}]"
)


def _parse_json_array(text: str) -> Optional[list]:
    """Extract the first JSON array from VLM output. Tolerates code fences."""
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def load_model(model_id: str, device: str, load_in_8bit: bool, max_pixels: int):
    """Load Qwen3-VL processor + model. `max_pixels` caps vision-token count."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    logger.info(f"Loading VLM: {model_id} (8bit={load_in_8bit}, max_pixels={max_pixels})")

    kwargs = {}
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = device
    else:
        kwargs["dtype"] = torch.bfloat16

    # max_pixels caps the per-image vision-token budget; Qwen-VL family
    # otherwise emits ~35K tokens for a 3840x2160 slide and OOMs on 16 GB.
    processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    if not load_in_8bit:
        model = model.to(device)
    model.eval()
    logger.info(f"VLM loaded onto {device}")
    return processor, model


def run_slide(processor, model, image_path: Path, max_new_tokens: int = 2048,
              resize_max_edge: int = 1280) -> dict:
    """Run one slide through the VLM. Returns dict with raw text + parsed JSON."""
    import torch
    from PIL import Image as PILImage

    pil = PILImage.open(image_path).convert("RGB")
    if resize_max_edge and max(pil.size) > resize_max_edge:
        pil.thumbnail((resize_max_edge, resize_max_edge), PILImage.LANCZOS)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    gen = outputs[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(gen, skip_special_tokens=True).strip()
    elapsed = time.time() - t0

    parsed = _parse_json_array(raw)
    return {
        "image": str(image_path),
        "elapsed_s": round(elapsed, 1),
        "raw_text": raw,
        "parsed": parsed,
        "parse_ok": parsed is not None,
        "n_regions": len(parsed) if parsed is not None else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL baseline probe on held-out slides")
    parser.add_argument("--images", nargs="*", default=None,
                        help="Slide image paths to run on. Default: 4 held-out Extremwertberechnungen pages.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/jakob_vlm_baseline"))
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-8bit", action="store_true",
                        help="Disable 8-bit loading (uses bf16, needs more VRAM).")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024,
                        help="Per-image pixel cap for the processor (≈1M pixels keeps vision "
                             "tokens under ~6K and fits 16 GB at 8-bit).")
    parser.add_argument("--resize-max-edge", type=int, default=1280,
                        help="Downscale input PIL image so max edge <= this many pixels before "
                             "passing to the processor. Belt-and-suspenders for --max-pixels.")
    args = parser.parse_args()

    image_paths = [Path(p) for p in (args.images or DEFAULT_HELD_OUT)]
    for p in image_paths:
        if not p.exists():
            logger.error(f"Missing slide: {p}")
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor, model = load_model(args.model_id, args.device,
                                  load_in_8bit=not args.no_8bit,
                                  max_pixels=args.max_pixels)

    summary = {
        "model_id": args.model_id,
        "device": args.device,
        "load_in_8bit": not args.no_8bit,
        "max_new_tokens": args.max_new_tokens,
        "results": [],
    }

    for img_path in image_paths:
        logger.info(f"Processing {img_path.name}")
        try:
            res = run_slide(processor, model, img_path,
                            max_new_tokens=args.max_new_tokens,
                            resize_max_edge=args.resize_max_edge)
        except Exception as e:
            logger.exception(f"VLM failed on {img_path}: {e}")
            res = {"image": str(img_path), "error": str(e)}
        stem = img_path.stem
        out_json = args.output_dir / f"{stem}.json"
        out_raw = args.output_dir / f"{stem}.raw.txt"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        if "raw_text" in res:
            with open(out_raw, "w", encoding="utf-8") as f:
                f.write(res["raw_text"])
        summary["results"].append({
            "image": str(img_path),
            "elapsed_s": res.get("elapsed_s"),
            "parse_ok": res.get("parse_ok"),
            "n_regions": res.get("n_regions"),
        })
        logger.info(f"  -> {out_json}  elapsed={res.get('elapsed_s')}s  "
                    f"regions={res.get('n_regions')}  parse_ok={res.get('parse_ok')}")

    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Done. Summary -> {args.output_dir / 'summary.json'}")

    del model
    del processor
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
