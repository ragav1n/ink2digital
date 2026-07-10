#!/usr/bin/env bash
# Overnight end-to-end batch driver: runs infer.py on every lecture under
# data/jakob_full_corpus/ with the canonical config (v3 detector + Jakob LoRA
# adapter + math OCR). Idempotent — re-running picks up where it left off.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source venv/bin/activate

CORPUS_DIR="data/jakob_full_corpus"
OUT_ROOT="outputs/jakob_full_corpus"
LOG_ROOT="logs/jakob_full_corpus"
STATUS="$LOG_ROOT/STATUS.txt"

DETECTOR="runs/detect/runs/jakob_detector_v3/hires1536-3/weights/best.pt"
ADAPTER="checkpoint/vlm_jakob_lora/best"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
echo "$(date -Is) BATCH START (driver pid $$)" | tee -a "$STATUS"

shopt -s nullglob
for lecture_dir in "$CORPUS_DIR"/*/; do
  slug="$(basename "$lecture_dir")"
  out_dir="$OUT_ROOT/$slug"
  log_file="$LOG_ROOT/${slug}.log"

  if [[ -s "$out_dir/results.json" ]]; then
    echo "$(date -Is) SKIP  $slug (already has results.json)" | tee -a "$STATUS"
    continue
  fi

  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  echo "$(date -Is) START $slug" | tee -a "$STATUS"

  python -u infer.py \
    --detector-path "$DETECTOR" \
    --detector-imgsz 960 \
    --image-dir "$lecture_dir" \
    --output-dir "$out_dir" \
    --ocr-backend vlm \
    --vlm-adapter "$ADAPTER" \
    --enable-math-ocr \
    --save-json \
    --device cuda \
    > "$log_file" 2>&1
  rc=$?

  echo "$(date -Is) DONE  $slug (exit $rc)" | tee -a "$STATUS"
done

echo "$(date -Is) BATCH COMPLETE" | tee -a "$STATUS"
