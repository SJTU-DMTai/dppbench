#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=1

TASK_DIR="${TASK_DIR:-$PWD/dppbench/tasks/berka}"
DATA_DIR="${DATA_DIR:-/mnt/sdb/dengjiale/dppbench/data}"
OUT_DIR="${OUT_DIR:-$PWD/logs/berka_diffprep_ablation_seed43}"

SEED="${SEED:-43}"
PYTHON="${PYTHON:-python}"
SCRIPT="scripts/ablations/berka_ablation.py"

if [[ ! -f "$TASK_DIR/model.yaml" || ! -f "$TASK_DIR/prepare.yaml" ]]; then
  echo "ERROR: task dir incorrect: $TASK_DIR (set TASK_DIR=...)">&2
  exit 2
fi
mkdir -p "$OUT_DIR"
fail=0
for case in order_full order_no_date order_no_freq order_no_date_no_freq account_full account_no_date account_no_freq account_no_date_no_freq; do
  echo "==== $case ===="
  args=(--case "$case" --task-dir "$TASK_DIR" --out "$OUT_DIR" --seed "$SEED")
  [[ -n "${DATA_DIR:-}" ]] && args+=(--data-dir "$DATA_DIR")
  [[ -n "${SOURCE_YAML:-}" ]] && args+=(--source-yaml "$SOURCE_YAML")
  "$PYTHON" "$SCRIPT" "${args[@]}" >"$OUT_DIR/$case.log" 2>&1 || { echo "FAILED: $case; see $OUT_DIR/$case.log"; fail=1; }
done
"$PYTHON" scripts/ablations/summarize_ablation.py --out "$OUT_DIR"
exit "$fail"
