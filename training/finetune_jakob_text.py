"""
Fine-tune Reptile-adapted TrOCR on Jakob's German handwriting.

Starts from `checkpoint/maml_ocr/meta_checkpoint_best.pt` (Reptile, val CER 0.47% on IAM)
and adapts to 50 Jakob crops (40 train / 10 val, seed 42).

Output: `checkpoint/trocr_jakob/best/` (HF format, ready for infer.py).

The fine-tune is only worth shipping if val CER beats Reptile on the SAME val split —
this script only writes the checkpoint; the 3-way comparison is in
scripts/eval_jakob_finetune.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup,
)

from utils.image_utils import load_image
from utils.metrics import batch_cer


class JakobLineDataset(Dataset):
    """Flat list JSON: [{"image": "jakob_finetune/text_001.jpg", "text": "..."}]"""

    def __init__(
        self,
        manifest_path: Path,
        processor: TrOCRProcessor,
        augment: bool = False,
        max_length: int = 128,
        data_root: Path = Path('data'),
    ):
        with open(manifest_path, encoding='utf-8') as f:
            self.samples = json.load(f)
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
        img_path = self.data_root / sample['image']

        try:
            img = load_image(img_path, mode='rgb')
            if self.augment:
                from utils.image_utils import augment_handwriting
                img = augment_handwriting(img)
            pil_img = Image.fromarray(img)
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            pil_img = Image.new('RGB', (384, 64), color=255)

        pixel_values = self.processor(
            images=pil_img, return_tensors='pt'
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text, return_tensors='pt', max_length=self.max_length,
            padding='max_length', truncation=True,
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {'pixel_values': pixel_values, 'labels': labels, 'text': text}


def collate(batch: List[Dict]) -> Dict:
    return {
        'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'texts': [b['text'] for b in batch],
    }


def load_reptile_init(
    processor_dir: Path,
    reptile_ckpt: Path,
    device: str,
) -> tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
    """Build TrOCR with the same architecture as the Reptile base, then overlay Reptile weights."""
    processor = TrOCRProcessor.from_pretrained(str(processor_dir))
    model = VisionEncoderDecoderModel.from_pretrained(str(processor_dir))

    ckpt = torch.load(str(reptile_ckpt), map_location='cpu', weights_only=False)
    state = ckpt['meta_model_state']
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading Reptile state: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    logger.info(f"Loaded Reptile weights from {reptile_ckpt} (epoch={ckpt.get('epoch')}, IAM val CER={ckpt.get('val_cer'):.4f})")

    # Generation config (mirrors finetune_german_ocr.py)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_new_tokens = 128
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4

    model.to(device)
    return model, processor


def freeze_encoder(model: VisionEncoderDecoderModel) -> int:
    """Freeze the vision encoder. Returns count of remaining trainable params."""
    for p in model.encoder.parameters():
        p.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Encoder frozen. Trainable: {n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M ({100*n_trainable/n_total:.1f}%)")
    return n_trainable


@torch.no_grad()
def evaluate(model, processor, loader, device) -> float:
    model.eval()
    hyps, refs = [], []
    for batch in loader:
        pixel_values = batch['pixel_values'].to(device)
        gen = model.generate(pixel_values)
        hyps.extend(processor.batch_decode(gen, skip_special_tokens=True))
        refs.extend(batch['texts'])
    return batch_cer(hyps, refs)['mean_cer']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=Path, default=Path('data/jakob_finetune/train.json'))
    p.add_argument('--val', type=Path, default=Path('data/jakob_finetune/val.json'))
    p.add_argument('--processor-dir', type=Path, default=Path('checkpoint/trocr_german/best'),
                   help='Source for TrOCR processor + arch')
    p.add_argument('--reptile-ckpt', type=Path,
                   default=Path('checkpoint/maml_ocr/meta_checkpoint_best.pt'))
    p.add_argument('--output-dir', type=Path, default=Path('checkpoint/trocr_jakob'))
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--warmup-ratio', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--unfreeze-encoder', action='store_true',
                   help='Train the full model (default: encoder frozen).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_reptile_init(args.processor_dir, args.reptile_ckpt, device)
    if not args.unfreeze_encoder:
        freeze_encoder(model)

    train_ds = JakobLineDataset(args.train, processor, augment=True)
    val_ds = JakobLineDataset(args.val, processor, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0, collate_fn=collate)

    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs // args.grad_accum)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    # Baseline (Reptile, pre-fine-tune) CER on Jakob val — this is what we must beat.
    reptile_val_cer = evaluate(model, processor, val_loader, device)
    logger.info(f"Pre-fine-tune Reptile val CER on Jakob: {reptile_val_cer:.4f} ({reptile_val_cer*100:.2f}%)")

    best_cer = reptile_val_cer
    no_improve = 0
    log = [{'epoch': 0, 'phase': 'reptile_init', 'val_cer': reptile_val_cer}]

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / args.grad_accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            total_loss += float(loss) * args.grad_accum

        train_loss = total_loss / max(len(train_loader), 1)
        val_cer = evaluate(model, processor, val_loader, device)
        is_best = val_cer < best_cer
        beats_reptile = val_cer < reptile_val_cer

        if is_best:
            best_cer = val_cer
            no_improve = 0
            hf_dir = args.output_dir / 'best'
            model.save_pretrained(str(hf_dir))
            processor.save_pretrained(str(hf_dir))
            logger.info(f"Saved best -> {hf_dir}")
        else:
            no_improve += 1

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"loss {train_loss:.4f} | val CER {val_cer:.4f} ({val_cer*100:.2f}%) | "
            f"{'BEST ' if is_best else f'no-improve {no_improve}/{args.patience} '}"
            f"{'BEATS-REPTILE' if beats_reptile else 'still-below-reptile'}"
        )
        log.append({
            'epoch': epoch + 1, 'train_loss': train_loss, 'val_cer': val_cer,
            'is_best': is_best, 'beats_reptile': beats_reptile,
        })

        if no_improve >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    log.append({'final': True, 'reptile_init_val_cer': reptile_val_cer,
                'best_val_cer': best_cer,
                'beats_reptile': best_cer < reptile_val_cer})
    with open(args.output_dir / 'training_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    logger.info(f"Done. Reptile-init val CER {reptile_val_cer:.4f} -> best {best_cer:.4f} "
                f"({'IMPROVED' if best_cer < reptile_val_cer else 'NO IMPROVEMENT'}).")


if __name__ == '__main__':
    main()
