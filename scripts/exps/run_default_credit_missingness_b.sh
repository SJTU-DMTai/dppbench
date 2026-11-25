#!/usr/bin/env bash
# Isolated B-layer experiments. Does not source common.sh or use its data root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON="${PYTHON:-python}"
SUITE_ROOT="${SUITE_ROOT:-/mnt/sdb/jiaqinyi/experiments/default_credit_missingness_pilot_v1}"
SCENARIO="${SCENARIO:-B_Mixed_q30_mask1001}"
SEED_LIST="${SEED_LIST:-42 43 44}"
BASELINE_LIST="${BASELINE_LIST:-Random SAGA AlphaClean Learn2Clean AutoPrep DAGPipe DiffPrep}"
GPU_ID="${GPU_ID:--1}"
HARD_MEMORY_GIB="${HARD_MEMORY_GIB:-85}"
MAX_MEMORY_MB="${MAX_MEMORY_MB:-71680}"
MAX_GPU_MEMORY_MB="${MAX_GPU_MEMORY_MB:-81920}"
MAX_EVALUATIONS="${MAX_EVALUATIONS:-1000}"
DRY_RUN="${DRY_RUN:-0}"
RETRY_FAILED="${RETRY_FAILED:-0}"
BATCH_ID="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="${LOG_DIR:-${SUITE_ROOT}/logs/${SCENARIO}/${BATCH_ID}}"
mkdir -p "$LOG_DIR"
read -r -a seeds <<< "$SEED_LIST"
read -r -a baselines <<< "$BASELINE_LIST"
FAILED=()

for seed in "${seeds[@]}"; do
  mkdir -p "${LOG_DIR}/seed${seed}"
  for baseline in "${baselines[@]}"; do
    LOG="${LOG_DIR}/seed${seed}/default_credit_${SCENARIO}_${baseline}_seed${seed}.log"
    if [[ -e "$LOG" ]]; then
      echo "[ERROR] Refusing to overwrite log: $LOG" >&2
      exit 2
    fi
    cmd=("$PYTHON" scripts/default_credit_missingness.py baselines
      --output-root "$SUITE_ROOT" --scenarios "$SCENARIO"
      --baselines "$baseline" --search-seeds "$seed" --gpu-id "$GPU_ID"
      --max-evaluations "$MAX_EVALUATIONS" --max-memory-mb "$MAX_MEMORY_MB"
      --max-gpu-memory-mb "$MAX_GPU_MEMORY_MB" --final-hpo --tee-log)
    if [[ "$RETRY_FAILED" == 1 ]]; then cmd+=(--retry-failed); fi
    if [[ "$DRY_RUN" != 1 ]]; then
      cmd+=(--execute)
      cmd=("$PYTHON" scripts/run_memory_limited.py --memory-gib "$HARD_MEMORY_GIB" -- "${cmd[@]}")
    fi
    echo "[START] scenario=$SCENARIO baseline=$baseline seed=$seed"
    echo "[LOG] $LOG"
    # Header, complete live child output, and exit status share one log.
    {
      echo "[TIME] $(date -Is)"
      echo "[SCENARIO] $SCENARIO"
      printf '[COMMAND]'; printf ' %q' "${cmd[@]}"; printf '\n'
    } > "$LOG"
    if "${cmd[@]}" >> "$LOG" 2>&1; then
      echo "[SUCCESS] baseline=$baseline seed=$seed" | tee -a "$LOG"
      if [[ "$DRY_RUN" != 1 ]]; then
        # Include the full CSV row (AUC, HPO, search-cost fields, etc.) in
        # the text log too. The canonical structured CSV remains unchanged.
        RESULT_ROOT="${SUITE_ROOT}/results/baselines/${SCENARIO}/seed${seed}/${baseline}"
        if "$PYTHON" - "$RESULT_ROOT" >> "$LOG" 2>&1 <<'PY'
import csv, json, sys
from pathlib import Path
files = sorted(Path(sys.argv[1]).rglob("results.csv"))
if not files:
    raise SystemExit("Missing expected results.csv; inspect run.log")
for path in files:
    print("[RESULTS_CSV]", path)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            print(json.dumps(row, ensure_ascii=False, indent=2))
PY
        then
          :
        else
          FAILED+=("$baseline seed=$seed result-summary-failed")
          echo "[FAILED] Could not append CSV summary: $LOG"
        fi
      fi
    else
      code=$?
      echo "[FAILED] baseline=$baseline seed=$seed exit=$code" | tee -a "$LOG"
      FAILED+=("$baseline seed=$seed exit=$code")
    fi
    echo "[END TIME] $(date -Is)" >> "$LOG"
  done
done

echo "[ALL DONE] $(date -Is)"
echo "[LOG_DIR] $LOG_DIR"
echo "[FAILED COUNT] ${#FAILED[@]}"
if ((${#FAILED[@]} > 0)); then
  printf '[FAILED] %s\n' "${FAILED[@]}"
  exit 1
fi
