"""
Phase 2 — Per-professor QLoRA adaptation of the Qwen3-VL OCR backend.

Phase 1 established that Qwen3-VL-8B zero-shot already reaches 2.04% CER on
Jakob handwriting (4.3x better than the TrOCR fine-tune). Phase 2 asks whether
*our* parameter-efficient adaptation can push it further — i.e. whether the VLM
is a component of our method, not just a borrowed model.

This trains a LoRA adapter on the language-model projections of Qwen3-VL-8B
(4-bit NF4 base, QLoRA-style) over the 40 Jakob line crops in
`data/jakob_finetune/train.json`. The base weights are frozen; only the adapter
(~0.1% of parameters) is learned, so the per-professor delta is a small file
that can be swapped in via `VLMOCRBackend(adapter_path=...)`.

The same training loop is the inner loop of a Reptile outer loop: with crops
tagged by lecture or professor, `--reptile` meta-learns the adapter
initialisation across tasks. With a single professor it reduces to plain
fine-tuning, which is what this script runs by default.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python training/finetune_vlm_jakob.py \
        --train data/jakob_finetune/train.json \
        --output checkpoint/vlm_jakob_lora \
        --epochs 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from loguru import logger
from PIL import Image

# Same transcription prompt the inference backend uses, so train/test match.
from models.vlm_ocr import _TEXT_PROMPT

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MAX_PIXELS = 1024 * 1024
RESIZE_MAX_EDGE = 1280


def load_crop(image_rel: str, data_root: Path) -> Image.Image:
    """Load a crop and downscale it the same way VLMOCRBackend does at inference."""
    pil = Image.open(data_root / image_rel).convert("RGB")
    if max(pil.size) > RESIZE_MAX_EDGE:
        pil.thumbnail((RESIZE_MAX_EDGE, RESIZE_MAX_EDGE), Image.LANCZOS)
    return pil


def build_sample(processor, pil: Image.Image, target: str, device: str):
    """Build a single supervised example: input_ids over [prompt + target],
    with the prompt span masked to -100 so loss is only on the transcription."""
    full_msgs = [
        {"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": _TEXT_PROMPT},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]
    full = processor.apply_chat_template(
        full_msgs, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=False,
    )
    # Prompt-only pass to locate the boundary between prompt and target tokens.
    prompt = processor.apply_chat_template(
        full_msgs[:1], tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True,
    )
    plen = prompt["input_ids"].shape[1]

    labels = full["input_ids"].clone()
    labels[:, :plen] = -100  # supervise only the assistant transcription
    full = {k: v.to(device) for k, v in full.items()}
    full["labels"] = labels.to(device)
    return full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=Path("data/jakob_finetune/train.json"))
    p.add_argument("--val", type=Path, default=Path("data/jakob_finetune/val.json"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("checkpoint/vlm_jakob_lora"))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)

    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    with open(args.train, encoding="utf-8") as f:
        train_data = json.load(f)
    logger.info(f"Training samples: {len(train_data)}")

    logger.info(f"Loading {MODEL_ID} in 4-bit (QLoRA base)...")
    # V100 (Volta) has no native bf16 — use fp16 compute for the 4-bit base.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map=device,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA on the language-model projections only — leave the vision tower frozen.
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.train()

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )

    n = len(train_data)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n).tolist()
        running, n_steps = 0.0, 0
        optim.zero_grad()
        for i, idx in enumerate(perm):
            s = train_data[idx]
            try:
                pil = load_crop(s["image"], args.data_root)
                batch = build_sample(processor, pil, s["text"], device)
                out = model(**batch)
                loss = out.loss / args.grad_accum
                loss.backward()
            except Exception as e:
                logger.warning(f"  skip sample {s.get('image')}: {e}")
                continue
            running += out.loss.item()
            n_steps += 1
            if (i + 1) % args.grad_accum == 0 or (i + 1) == n:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                optim.zero_grad()
        mean_loss = running / max(n_steps, 1)
        logger.info(f"Epoch {epoch}/{args.epochs}  mean loss = {mean_loss:.4f}")
        if mean_loss < best_loss:
            best_loss = mean_loss
            model.save_pretrained(str(args.output / "best"))
            logger.info(f"  saved best adapter -> {args.output / 'best'} "
                        f"(loss {best_loss:.4f})")

    model.save_pretrained(str(args.output / "last"))
    with open(args.output / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_id": MODEL_ID,
            "train_samples": n,
            "epochs": args.epochs,
            "lr": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "best_train_loss": best_loss,
        }, f, indent=2)
    logger.info(f"Done. Adapter at {args.output}/best  (best train loss {best_loss:.4f})")


if __name__ == "__main__":
    main()
