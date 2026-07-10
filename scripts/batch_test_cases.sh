#!/usr/bin/env bash
# Run infer.py on each curated demo slide under test_cases/.
# Mirrors scripts/batch_e2e_jakob.sh: same detector, same VLM adapter, same flags
# so test_cases results are directly comparable to the full corpus run.
# Idempotent — re-running picks up where it left off (skips slots that already
# have a results.json).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source venv/bin/activate

CASES_DIR="test_cases"
OUT_ROOT="outputs/test_cases"
LOG_ROOT="logs/test_cases"
STATUS="$LOG_ROOT/STATUS.txt"

DETECTOR="runs/detect/runs/jakob_detector_v3/hires1536-3/weights/best.pt"
ADAPTER="checkpoint/vlm_jakob_lora/best"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
echo "$(date -Is) BATCH START (driver pid $$)" | tee -a "$STATUS"

shopt -s nullglob
for case_dir in "$CASES_DIR"/[0-9][0-9]_*/; do
  slug="$(basename "$case_dir")"
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
    --image-dir "$case_dir" \
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
