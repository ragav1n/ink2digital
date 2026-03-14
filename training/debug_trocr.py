"""
Quick debug script: run 2 batches of TrOCR training with num_workers=0
to surface any silent crashes. Run from project root:
    python training/debug_trocr.py
"""
import sys
import traceback
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.finetune_german_ocr import GermanHandwritingDataset, collate_fn

MANIFEST = "data/processed/german_text/german_text_train.json"
MODEL_ID  = "microsoft/trocr-large-handwritten"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BATCH     = 2
MAX_LEN   = 128

logger.info(f"Device: {DEVICE} | PyTorch: {torch.__version__}")

try:
    logger.info("Loading processor + model…")
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.to(DEVICE)
    logger.info("Model loaded OK")

    logger.info("Building dataset…")
    ds = GermanHandwritingDataset(MANIFEST, processor, augment=False, max_length=MAX_LEN)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    logger.info("Running 3 training steps…")
    model.train()
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        if step >= 3:
            break
        logger.info(f"  Step {step}: pixel_values={batch['pixel_values'].shape}, labels={batch['labels'].shape}")
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels       = batch["labels"].to(DEVICE)

        try:
            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                logger.info(f"  Step {step}: loss={loss.item():.4f}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                logger.info(f"  Step {step}: loss={loss.item():.4f}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
        except Exception:
            logger.error(f"CRASH at step {step}:")
            traceback.print_exc()
            sys.exit(1)

    logger.info("3 steps completed successfully — training should work.")
    logger.info("Now try: nohup python training/finetune_german_ocr.py --train-data data/processed/german_text/german_text_train.json --val-data data/processed/german_text/german_text_val.json --output-dir checkpoints/trocr_german --epochs 30 --batch 8 --workers 0 --device cuda > logs/trocr_finetune.log 2>&1 &")

except Exception:
    logger.error("Outer crash:")
    traceback.print_exc()
    sys.exit(1)
