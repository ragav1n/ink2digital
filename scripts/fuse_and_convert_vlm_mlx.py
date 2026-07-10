"""Fuse the Jakob LoRA adapter into Qwen3-VL and convert it to an MLX 8-bit model.

The Phase-2 per-professor LoRA adapter (`checkpoint/vlm_jakob_lora/`) is a PEFT
adapter on the `transformers` base model — MLX cannot load it directly. This
one-time script bakes it in:

  1. load base Qwen3-VL-8B + the PEFT adapter, `merge_and_unload()`
  2. save the merged fp16 model
  3. convert the merged model to an MLX 8-bit checkpoint

The result is a self-contained MLX model directory. Point the pipeline at it:

    python infer.py ... --ocr-backend vlm --device mps \\
        --vlm-mlx-model models_mlx/qwen3vl-8b-jakob-mlx-8bit

Quality: base model ~2.04% Jakob val CER, fused ~1.46%. If this conversion
fails, the pipeline still works with the base model (omit --vlm-mlx-model).

Run once, OFFLINE-PREP in Germany. Extra deps beyond requirements-mac.txt:
    pip install peft
Needs ~16 GB free RAM to hold the fp16 model during the merge.

Usage:
    python scripts/fuse_and_convert_vlm_mlx.py \\
        --base Qwen/Qwen3-VL-8B-Instruct \\
        --adapter checkpoint/vlm_jakob_lora/best \\
        --out models_mlx/qwen3vl-8b-jakob-mlx-8bit
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def merge_adapter(base: str, adapter: str, merged_dir: Path) -> None:
    """Load base + PEFT adapter, merge, and save the fp16 model + processor."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[1/3] Loading base model: {base}")
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.float16)

    print(f"[2/3] Applying + merging LoRA adapter: {adapter}")
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir))
    # The processor must travel with the model for MLX conversion / inference.
    AutoProcessor.from_pretrained(base).save_pretrained(str(merged_dir))
    print(f"      merged fp16 model saved -> {merged_dir}")


def restore_config_defaults(base: str, merged_dir: Path) -> None:
    """Backfill config fields that ``save_pretrained`` dropped as defaults.

    transformers writes a diff-only config (omitting values equal to the class
    defaults), but mlx_vlm's config dataclasses require some of them explicitly
    (e.g. Qwen3-VL ``text_config.rope_theta``), so the convert step crashes with
    a missing-argument TypeError. We read the base's *original* config.json
    (which spells those fields out) and add back any key the merged config is
    missing — never overwriting merged values. NB: ``AutoConfig.to_dict()`` is
    itself diff-only here, so we must read the raw file, not a live config.
    """
    import json
    import os

    if os.path.isdir(base):
        base_cfg_path = os.path.join(base, 'config.json')
    else:
        from huggingface_hub import hf_hub_download
        base_cfg_path = hf_hub_download(base, 'config.json')  # cached, no re-download
    full = json.loads(Path(base_cfg_path).read_text())
    cfg_path = merged_dir / 'config.json'
    merged = json.loads(cfg_path.read_text())

    def fill(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                fill(dst[k], v)
            elif k not in dst:
                dst[k] = v

    fill(merged, full)

    # transformers names the sub-config model_types 'qwen3_vl_vision' /
    # 'qwen3_vl_text', but mlx_vlm's qwen3_vl VisionModel only accepts
    # {'qwen3_vl','qwen3_5','qwen3_5_moe'}. Normalise so the convert succeeds.
    vc = merged.get('vision_config')
    if isinstance(vc, dict) and vc.get('model_type', '').startswith('qwen3_vl'):
        vc['model_type'] = 'qwen3_vl'

    cfg_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"      fixed config for mlx_vlm (rope_theta + vision model_type) -> {cfg_path}")


def convert_to_mlx(merged_dir: Path, out_dir: Path, q_bits: int = 8) -> None:
    """Convert a HF model directory to an MLX quantized checkpoint."""
    print(f"[3/3] Converting to MLX {q_bits}-bit -> {out_dir}")
    # The CLI is more stable across mlx-vlm releases than the Python API.
    # If this errors, check `python -m mlx_vlm convert --help` for the flags
    # in the installed version.
    cmd = [
        sys.executable, '-m', 'mlx_vlm', 'convert',
        '--hf-path', str(merged_dir),
        '--mlx-path', str(out_dir),
        '-q', '--q-bits', str(q_bits),
    ]
    print('      ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default='Qwen/Qwen3-VL-8B-Instruct',
                    help='Base HF model id (or local path).')
    ap.add_argument('--adapter', default='checkpoint/vlm_jakob_lora/best',
                    help='PEFT LoRA adapter directory.')
    ap.add_argument('--out', type=Path, default=Path('models_mlx/qwen3vl-8b-jakob-mlx-8bit'),
                    help='Output directory for the MLX model.')
    ap.add_argument('--q-bits', type=int, default=8, choices=(4, 6, 8),
                    help='MLX quantization bits (default 8 = matches the server).')
    ap.add_argument('--keep-merged', action='store_true',
                    help='Keep the intermediate fp16 merged model.')
    args = ap.parse_args()

    if not Path(args.adapter).exists():
        ap.error(f"adapter not found: {args.adapter}")

    merged_dir = (Path(args.keep_merged and (str(args.out) + '_merged_fp16')
                       or tempfile.mkdtemp(prefix='qwen3vl_merged_')))
    try:
        merge_adapter(args.base, args.adapter, merged_dir)
        restore_config_defaults(args.base, merged_dir)
        convert_to_mlx(merged_dir, args.out, q_bits=args.q_bits)
        print(f"\nDone. MLX model -> {args.out}")
        print(f"Use it:  python infer.py ... --ocr-backend vlm --device mps "
              f"--vlm-mlx-model {args.out}")
    finally:
        if not args.keep_merged and merged_dir.exists():
            shutil.rmtree(merged_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
