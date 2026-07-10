"""
Honest 3-way comparison on Jakob val split (10 samples, seed=42):
    1. Base TrOCR fine-tuned on IAM German (checkpoint/trocr_german/best)
    2. Reptile meta-init (checkpoint/maml_ocr/meta_checkpoint_best.pt)
    3. Jakob fine-tune from Reptile init (checkpoint/trocr_jakob/best)

The Jakob fine-tune is only worth shipping if it beats Reptile on this split.
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
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from utils.image_utils import load_image
from utils.metrics import batch_cer


def load_model(path: Path, device: str) -> tuple:
    """Load an HF-format TrOCR checkpoint."""
    processor = TrOCRProcessor.from_pretrained(str(path))
    model = VisionEncoderDecoderModel.from_pretrained(str(path)).to(device).eval()
    return model, processor


def load_reptile(processor_dir: Path, ckpt: Path, device: str) -> tuple:
    """Build TrOCR with the IAM-finetune arch, then overlay Reptile state."""
    processor = TrOCRProcessor.from_pretrained(str(processor_dir))
    model = VisionEncoderDecoderModel.from_pretrained(str(processor_dir))
    state = torch.load(str(ckpt), map_location='cpu', weights_only=False)['meta_model_state']
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), processor


@torch.no_grad()
def predict(model, processor, val_data: list, device: str) -> tuple[list[str], list[str]]:
    hyps, refs = [], []
    for sample in val_data:
        img = load_image(Path('data') / sample['image'], mode='rgb')
        pixel = processor(images=Image.fromarray(img), return_tensors='pt').pixel_values.to(device)
        gen = model.generate(pixel)
        hyps.append(processor.batch_decode(gen, skip_special_tokens=True)[0])
        refs.append(sample['text'])
    return hyps, refs


def predict_vlm(val_data: list, device: str,
                adapter_path: str | None = None) -> tuple[list[str], list[str]]:
    """Run the same val set through Qwen3-VL-8B.

    adapter_path=None  -> zero-shot base model.
    adapter_path set   -> with the Phase 2 per-professor LoRA adapter.
    Returns (hyps, refs) parallel to predict()."""
    from models.vlm_ocr import VLMOCRBackend
    backend = VLMOCRBackend(device=device, adapter_path=adapter_path)
    crops = [load_image(Path('data') / s['image'], mode='rgb') for s in val_data]
    hyps = backend.predict(crops, mode='text')
    refs = [s['text'] for s in val_data]
    backend.unload()
    return hyps, refs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--val', type=Path, default=Path('data/jakob_finetune/val.json'))
    p.add_argument('--output', type=Path, default=Path('outputs/jakob_finetune_eval.json'))
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--include-vlm', action=argparse.BooleanOptionalAction, default=True,
                   help='Also evaluate Qwen3-VL-8B (zero-shot) as a 4th row. Default ON.')
    p.add_argument('--vlm-adapter', type=str, default=None,
                   help='Path to a Phase 2 per-professor LoRA adapter. When set, '
                        'adds a 5th row: Qwen3-VL-8B + adapter.')
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.val) as f:
        val_data = json.load(f)

    models = [
        ('base_trocr_iam', lambda: load_model(Path('checkpoint/trocr_german/best'), device)),
        ('reptile_meta', lambda: load_reptile(
            Path('checkpoint/trocr_german/best'),
            Path('checkpoint/maml_ocr/meta_checkpoint_best.pt'),
            device,
        )),
        ('jakob_finetune', lambda: load_model(Path('checkpoint/trocr_jakob/best'), device)),
    ]

    results = {}
    for name, loader in models:
        logger.info(f"=== {name} ===")
        model, processor = loader()
        hyps, refs = predict(model, processor, val_data, device)
        metrics = batch_cer(hyps, refs)
        results[name] = {
            'mean_cer': metrics['mean_cer'],
            'predictions': [
                {'image': v['image'], 'ref': r, 'hyp': h, 'cer': c}
                for v, r, h, c in zip(val_data, refs, hyps, metrics['per_sample'])
            ],
        }
        logger.info(f"  Mean CER: {metrics['mean_cer']:.4f} ({metrics['mean_cer']*100:.2f}%)")
        for v, r, h, c in zip(val_data, refs, hyps, metrics['per_sample']):
            mark = '✓' if c < 0.05 else ('~' if c < 0.2 else '✗')
            logger.info(f"  {mark} CER={c:.3f}  REF: {r!r}")
            logger.info(f"              HYP: {h!r}")
        del model, processor
        torch.cuda.empty_cache()

    if args.include_vlm:
        logger.info("=== vlm_qwen3 ===")
        hyps, refs = predict_vlm(val_data, device)
        metrics = batch_cer(hyps, refs)
        results['vlm_qwen3'] = {
            'mean_cer': metrics['mean_cer'],
            'predictions': [
                {'image': v['image'], 'ref': r, 'hyp': h, 'cer': c}
                for v, r, h, c in zip(val_data, refs, hyps, metrics['per_sample'])
            ],
        }
        logger.info(f"  Mean CER: {metrics['mean_cer']:.4f} ({metrics['mean_cer']*100:.2f}%)")
        for v, r, h, c in zip(val_data, refs, hyps, metrics['per_sample']):
            mark = '✓' if c < 0.05 else ('~' if c < 0.2 else '✗')
            logger.info(f"  {mark} CER={c:.3f}  REF: {r!r}")
            logger.info(f"              HYP: {h!r}")

    if args.vlm_adapter:
        logger.info(f"=== vlm_qwen3_adapted ({args.vlm_adapter}) ===")
        hyps, refs = predict_vlm(val_data, device, adapter_path=args.vlm_adapter)
        metrics = batch_cer(hyps, refs)
        results['vlm_qwen3_adapted'] = {
            'mean_cer': metrics['mean_cer'],
            'adapter_path': args.vlm_adapter,
            'predictions': [
                {'image': v['image'], 'ref': r, 'hyp': h, 'cer': c}
                for v, r, h, c in zip(val_data, refs, hyps, metrics['per_sample'])
            ],
        }
        logger.info(f"  Mean CER: {metrics['mean_cer']:.4f} ({metrics['mean_cer']*100:.2f}%)")
        for v, r, h, c in zip(val_data, refs, hyps, metrics['per_sample']):
            mark = '✓' if c < 0.05 else ('~' if c < 0.2 else '✗')
            logger.info(f"  {mark} CER={c:.3f}  REF: {r!r}")
            logger.info(f"              HYP: {h!r}")

    # Summary table
    logger.info("\n=== SUMMARY ===")
    base = results['base_trocr_iam']['mean_cer']
    rep = results['reptile_meta']['mean_cer']
    jak = results['jakob_finetune']['mean_cer']
    logger.info(f"  Base TrOCR (IAM):       {base*100:6.2f}%")
    logger.info(f"  Reptile meta:           {rep*100:6.2f}%")
    logger.info(f"  Jakob fine-tune:        {jak*100:6.2f}%")
    if 'vlm_qwen3' in results:
        vlm = results['vlm_qwen3']['mean_cer']
        logger.info(f"  Qwen3-VL-8B (zero-shot): {vlm*100:6.2f}%")
        logger.info(f"  VLM vs Jakob FT:        {(jak-vlm)*100:+6.2f}pp "
                    f"({'VLM WINS' if vlm < jak else 'JAKOB FT WINS'})")
    if 'vlm_qwen3_adapted' in results:
        vlm_a = results['vlm_qwen3_adapted']['mean_cer']
        logger.info(f"  Qwen3-VL-8B + LoRA:      {vlm_a*100:6.2f}%")
        if 'vlm_qwen3' in results:
            vlm = results['vlm_qwen3']['mean_cer']
            logger.info(f"  Adapted vs zero-shot:   {(vlm-vlm_a)*100:+6.2f}pp "
                        f"({'ADAPTER WINS' if vlm_a < vlm else 'ZERO-SHOT WINS'})")
    logger.info(f"  Jakob vs Reptile:       {(rep-jak)*100:+6.2f}pp ({'WIN' if jak < rep else 'LOSS'})")

    results['summary'] = {
        'base_trocr_iam_cer': base,
        'reptile_cer': rep,
        'jakob_finetune_cer': jak,
        'vlm_qwen3_cer': results.get('vlm_qwen3', {}).get('mean_cer'),
        'vlm_qwen3_adapted_cer': results.get('vlm_qwen3_adapted', {}).get('mean_cer'),
        'jakob_beats_reptile': jak < rep,
        'absolute_improvement_vs_reptile_pp': (rep - jak) * 100,
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {args.output}")


if __name__ == '__main__':
    main()
