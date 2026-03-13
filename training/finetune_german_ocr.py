"""
Fine-tune TrOCR on German handwriting data (IAM German subset).
Phase 2 (Week 5): Improve CER from ~15-20% (English model) to ~5-8%.

This script fine-tunes microsoft/trocr-large-handwritten on the German IAM
data prepared by scripts/prepare_iam_german.py.

Architecture:
    TrOCR = BEiT (Vision Encoder) + RoBERTa (Language Decoder)
    We fine-tune the full model end-to-end on German handwriting.

Expected improvement:
    Before: CER ~15-20% (English-pretrained, no German fine-tuning)
    After:  CER ~5-8% (after German fine-tuning)
    Target: CER ~3-4% (after meta-learning in Phase 3)

Usage:
    python training/finetune_german_ocr.py \
        --train-data data/processed/german_text/german_text_train.json \
        --val-data data/processed/german_text/german_text_val.json \
        --output-dir checkpoints/trocr_german \
        --epochs 30 \
        --batch 8

    # Resume:
    python training/finetune_german_ocr.py \
        --resume checkpoints/trocr_german/checkpoint-best \
        --epochs 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup,
)

from utils.metrics import compute_cer, batch_cer
from utils.image_utils import load_image
from utils.german_postprocessing import batch_correct


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GermanHandwritingDataset(Dataset):
    """
    Dataset for German handwriting OCR.
    Loads from JSON manifests produced by scripts/prepare_iam_german.py.

    Args:
        manifest_path: Path to JSON file with 'samples' list.
        processor: TrOCR processor for image preprocessing.
        augment: Apply data augmentation (training only).
        max_length: Maximum target sequence length.
        data_root: Root for resolving relative image paths.
    """

    def __init__(
        self,
        manifest_path: Path,
        processor: TrOCRProcessor,
        augment: bool = False,
        max_length: int = 128,
        data_root: Path = Path('data'),
    ):
        with open(manifest_path, encoding='utf-8') as f:
            data = json.load(f)
        self.samples = data['samples']
        self.processor = processor
        self.augment = augment
        self.max_length = max_length
        self.data_root = data_root
        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        text = sample['text']

        # Load image
        img_path = Path(sample['image'])
        if not img_path.is_absolute():
            img_path = self.data_root / img_path

        try:
            img = load_image(img_path, mode='rgb')
            if self.augment:
                from utils.image_utils import augment_handwriting
                img = augment_handwriting(img)
            pil_img = Image.fromarray(img)
        except Exception as e:
            logger.debug(f"Failed to load {img_path}: {e}")
            # Return blank image as fallback
            pil_img = Image.new('RGB', (384, 64), color=255)

        # Process image
        pixel_values = self.processor(
            images=pil_img,
            return_tensors='pt',
        ).pixel_values.squeeze(0)

        # Process text labels
        with self.processor.as_target_processor():
            labels = self.processor(
                text,
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
            ).input_ids.squeeze(0)

        # Replace padding with -100 (ignored in loss)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'text': text,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate batch — handle variable-length sequences."""
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    texts = [b['text'] for b in batch]
    return {'pixel_values': pixel_values, 'labels': labels, 'texts': texts}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class GermanTrOCRTrainer:
    """
    Fine-tune TrOCR on German handwriting with:
    - AdamW optimizer
    - Cosine LR schedule with warmup
    - CER-based model selection
    - Gradient accumulation for effective larger batch size
    - AMP (automatic mixed precision) for RTX 4060 Ti
    """

    def __init__(
        self,
        model_id: str = 'microsoft/trocr-large-handwritten',
        output_dir: Path = Path('checkpoints/trocr_german'),
        device: str = 'cuda',
        learning_rate: float = 5e-5,
        num_epochs: int = 30,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_length: int = 128,
        eval_every: int = 1,
        save_best_only: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_length = max_length
        self.eval_every = eval_every
        self.save_best_only = save_best_only

        logger.info(f"Loading TrOCR: {model_id}")
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)

        # Configure decoder for German
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = max_length
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4

        self.model.to(self.device)
        logger.info(f"Model on {self.device}")

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

    def train(
        self,
        train_manifest: Path,
        val_manifest: Path,
        resume_from: Optional[str] = None,
    ) -> dict:
        """
        Run the fine-tuning loop.

        Args:
            train_manifest: Path to training JSON manifest.
            val_manifest: Path to validation JSON manifest.
            resume_from: Path to checkpoint to resume from.

        Returns:
            Final metrics dict.
        """
        # Datasets
        train_ds = GermanHandwritingDataset(
            train_manifest, self.processor, augment=True, max_length=self.max_length
        )
        val_ds = GermanHandwritingDataset(
            val_manifest, self.processor, augment=False, max_length=self.max_length
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=4, collate_fn=collate_fn, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size * 2, shuffle=False,
            num_workers=4, collate_fn=collate_fn, pin_memory=True,
        )

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        # Resume if needed
        start_epoch = 0
        best_cer = float('inf')
        if resume_from and Path(resume_from).exists():
            ckpt = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_cer = ckpt.get('best_cer', float('inf'))
            logger.info(f"Resumed from {resume_from} (epoch {start_epoch})")

        logger.info(f"Training {self.num_epochs} epochs on {len(train_ds)} samples")
        logger.info(f"Validation: {len(val_ds)} samples")

        training_log = []

        for epoch in range(start_epoch, self.num_epochs):
            # Training
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)

            # Evaluation
            if epoch % self.eval_every == 0:
                val_cer = self._evaluate(val_loader)
                is_best = val_cer < best_cer
                if is_best:
                    best_cer = val_cer

                logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val CER: {val_cer:.4f} ({val_cer*100:.2f}%) | "
                    f"{'⭐ Best' if is_best else ''}"
                )

                training_log.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_cer': val_cer,
                    'is_best': is_best,
                })

                # Save checkpoint
                self._save_checkpoint(epoch, optimizer, val_cer, best_cer, is_best)

        logger.info(f"Training complete. Best CER: {best_cer:.4f} ({best_cer*100:.2f}%)")

        # Save training log
        with open(self.output_dir / 'training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)

        return {'best_val_cer': best_cer, 'epochs_trained': self.num_epochs}

    def _train_epoch(self, loader: DataLoader, optimizer, scheduler) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_steps = 0

        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += float(loss) * self.gradient_accumulation_steps
            n_steps += 1

        return total_loss / max(n_steps, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        """Evaluate on validation set. Returns mean CER."""
        self.model.eval()
        all_hyps, all_refs = [], []

        for batch in loader:
            pixel_values = batch['pixel_values'].to(self.device)
            refs = batch['texts']

            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=self.max_length,
                num_beams=4,
            )
            hyps = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            all_hyps.extend(hyps)
            all_refs.extend(refs)

        result = batch_cer(all_hyps, all_refs)
        return result['mean_cer']

    def _save_checkpoint(
        self,
        epoch: int,
        optimizer,
        val_cer: float,
        best_cer: float,
        is_best: bool,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_cer': val_cer,
            'best_cer': best_cer,
        }

        if is_best or not self.save_best_only:
            # Save full checkpoint
            ckpt_path = self.output_dir / f'checkpoint-epoch-{epoch+1}.pt'
            if is_best:
                ckpt_path = self.output_dir / 'checkpoint-best.pt'
            torch.save(checkpoint, ckpt_path)

            # Also save HuggingFace format for easy loading
            if is_best:
                hf_dir = self.output_dir / 'best'
                self.model.save_pretrained(str(hf_dir))
                self.processor.save_pretrained(str(hf_dir))
                logger.info(f"Best model saved -> {hf_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on German handwriting")
    parser.add_argument('--model', type=str, default='microsoft/trocr-large-handwritten',
                        help='Base model ID or path')
    parser.add_argument('--train-data', type=Path,
                        default=Path('data/processed/german_text/german_text_train.json'))
    parser.add_argument('--val-data', type=Path,
                        default=Path('data/processed/german_text/german_text_val.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('checkpoints/trocr_german'))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--grad-accum', type=int, default=4,
                        help='Gradient accumulation steps')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    trainer = GermanTrOCRTrainer(
        model_id=args.model,
        output_dir=args.output_dir,
        device=args.device,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
    )

    metrics = trainer.train(
        train_manifest=args.train_data,
        val_manifest=args.val_data,
        resume_from=args.resume,
    )

    logger.info(f"Final metrics: {metrics}")
