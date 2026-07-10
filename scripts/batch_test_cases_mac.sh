#!/usr/bin/env bash
# Mac variant of scripts/batch_test_cases.sh — runs the curated test_cases/
# demo set on an Apple Silicon Mac using the MLX VLM backend.
#
# Use this on the M-series Mac for the offline India demo. On the V100 server,
# use scripts/batch_test_cases.sh instead.
#
# Required on the Mac BEFORE running (see MAC_SETUP.md for full setup):
#   1. venv-mac/ with requirements-mac.txt installed.
#   2. Detector weights at:
#        runs/detect/runs/jakob_detector_v3/hires1536-3/weights/best.pt
#      (scp from V100, ~few MB.)
#   3. The 10 source slide JPGs at:
#        test_cases/01_hero/slide.jpg
#        test_cases/02_handwritten_text/slide.jpg
#        ... through test_cases/10_failure_case/slide.jpg
#      (scp from V100 — these are gitignored as private slide imagery.)
#   4. MLX VLM model cached online once (~8.5 GB):
#        hf download lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit
#   5. (Optional, higher quality) Fused Jakob-MLX model built once via
#      scripts/fuse_and_convert_vlm_mlx.py — placed at:
#        models_mlx/qwen3vl-8b-jakob-mlx-8bit/
#      If present, this script uses it automatically; if absent, falls back
#      to the base 8-bit model.
#
# Idempotent — re-running skips slots that already have results.json.
set -uo pipefail
source venv-mac/bin/activate

CASES_DIR="test_cases"
OUT_ROOT="outputs/test_cases"
LOG_ROOT="logs/test_cases"
STATUS="$LOG_ROOT/STATUS.txt"

DETECTOR="runs/detect/runs/jakob_detector_v3/hires1536-3/weights/best.pt"
FUSED_MLX_MODEL="models_mlx/qwen3vl-8b-jakob-mlx-8bit"

if [[ -d "$FUSED_MLX_MODEL" ]]; then
  echo "Using fused Jakob-MLX model at $FUSED_MLX_MODEL"
  VLM_FLAGS=(--vlm-mlx-model "$FUSED_MLX_MODEL")
else
  echo "Fused Jakob-MLX model not found; using base MLX 8-bit model"
  VLM_FLAGS=()
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
echo "$(date -Iseconds) BATCH START (driver pid $$)" | tee -a "$STATUS"

shopt -s nullglob
for case_dir in "$CASES_DIR"/[0-9][0-9]_*/; do
  slug="$(basename "$case_dir")"
  out_dir="$OUT_ROOT/$slug"
  log_file="$LOG_ROOT/${slug}.log"

  if [[ -s "$out_dir/results.json" ]]; then
    echo "$(date -Iseconds) SKIP  $slug (already has results.json)" | tee -a "$STATUS"
    continue
  fi

  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  echo "$(date -Iseconds) START $slug" | tee -a "$STATUS"

  python -u infer.py \
    --detector-path "$DETECTOR" \
    --detector-imgsz 960 \
    --image-dir "$case_dir" \
    --output-dir "$out_dir" \
    --ocr-backend vlm \
    --device mps \
    "${VLM_FLAGS[@]}" \
    --enable-math-ocr \
    --save-json \
    > "$log_file" 2>&1
  rc=$?

  echo "$(date -Iseconds) DONE  $slug (exit $rc)" | tee -a "$STATUS"
done

echo "$(date -Iseconds) BATCH COMPLETE" | tee -a "$STATUS"
