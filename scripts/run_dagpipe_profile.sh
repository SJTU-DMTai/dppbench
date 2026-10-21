#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/jiaqinyi/code/dppbench}"
DATA_DIR="${DATA_DIR:-/mnt/sdb/dengjiale/dppbench/data}"
GPU_ID="${GPU_ID:-1}"
TASKS="${TASKS:-kuairec}"
VARIANT="${VARIANT:-optimized_auto}"
STEPS="${STEPS:-20}"
EPOCHS="${EPOCHS:-200}"
PROFILE_STEPS="${PROFILE_STEPS:-0}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/sdb/jiaqinyi/dppbench/output/dagpipe_profile}"
LOG_BASE="${LOG_BASE:-${REPO_ROOT}/logs/dagpipe_profile}"

cd "$REPO_ROOT"
[[ -f scripts/profile_dagpipe_performance.py ]] || { echo "Missing scripts/profile_dagpipe_performance.py" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "Missing data directory: $DATA_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_BASE" "$LOG_BASE"

run_id="$(date +%Y%m%d_%H%M%S)_$$"
printf '[env] repo=%s data=%s gpu=%s tasks=%s variant=%s steps=%s epochs=%s profile_steps=%s run_id=%s\n' \
  "$REPO_ROOT" "$DATA_DIR" "$GPU_ID" "$TASKS" "$VARIANT" "$STEPS" "$EPOCHS" "$PROFILE_STEPS" "$run_id"

for task in ${TASKS//,/ }; do
  case "$task" in kuairec|yelp) ;; *) echo "Unsupported task: $task" >&2; exit 2 ;; esac
  out="$OUTPUT_BASE/${task}_${VARIANT}_${run_id}"
  log="$LOG_BASE/${task}_${VARIANT}_${run_id}.log"
  echo "[START] task=$task output=$out log=$log time=$(date)"
  python -u scripts/profile_dagpipe_performance.py \
    --data_dir "$DATA_DIR" \
    --task "$task" \
    --gpu_id "$GPU_ID" \
    --output_dir "$out" \
    --steps "$STEPS" \
    --epochs "$EPOCHS" \
    --profile_steps "$PROFILE_STEPS" \
    --variant "$VARIANT" \
    2>&1 | tee "$log"
  echo "[DONE] task=$task time=$(date)"
done
