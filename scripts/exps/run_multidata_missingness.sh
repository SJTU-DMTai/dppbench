#!/usr/bin/env bash
# Additional datasets; unchanged original exp1 entry, configs and loaders.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PERTURBED_DATA_ROOT="${PERTURBED_DATA_ROOT:-/mnt/sdb/jiaqinyi/experiments/multidata_MCAR30_mask1001_v1}"
TASK_LIST="${TASK_LIST:-fraud_detection home_credit polish_bankruptcy}"
SEED_LIST="${SEED_LIST:-42}"
BASELINE_LIST="${BASELINE_LIST:-Random SAGA AlphaClean Learn2Clean AutoPrep DAGPipe DiffPrep}"
GPU_ID="${GPU_ID:-1}"
HARD_MEMORY_GIB="${HARD_MEMORY_GIB:-85}"
BATCH="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="${LOG_DIR:-logs/multidata_MCAR30_mask1001_${BATCH}}"
NEW_OUTPUT_ROOT="${NEW_OUTPUT_ROOT:-${PERTURBED_DATA_ROOT}/runs/${BATCH}}"
if [[ -e "$NEW_OUTPUT_ROOT" ]]; then echo "[ERROR] Existing output root: $NEW_OUTPUT_ROOT" >&2; exit 2; fi
python scripts/verify_multidata_batch.py --output-root "$PERTURBED_DATA_ROOT"
read -r -a tasks <<< "$TASK_LIST"
read -r -a seeds <<< "$SEED_LIST"
read -r -a baselines <<< "$BASELINE_LIST"
FAILED=()
for seed in "${seeds[@]}"; do
  mkdir -p "$LOG_DIR/seed${seed}"
  for task in "${tasks[@]}"; do
    for baseline in "${baselines[@]}"; do
      LOG="$LOG_DIR/seed${seed}/${task}_${baseline}_seed${seed}.log"
      if [[ -e "$LOG" ]]; then echo "[ERROR] Existing log: $LOG" >&2; exit 2; fi
      echo "[START] task=$task baseline=$baseline seed=$seed log=$LOG"
      if TASKS="$task" SEEDS="$seed" BASELINES="$baseline" \
         DATA_DIR="$PERTURBED_DATA_ROOT" OUTPUT_ROOT="$NEW_OUTPUT_ROOT" GPU_ID="$GPU_ID" \
         python scripts/run_memory_limited.py --memory-gib "$HARD_MEMORY_GIB" -- \
         bash scripts/exps/01_exp1_1_cost_effectiveness.sh > "$LOG" 2>&1; then
        echo "[SUCCESS] task=$task baseline=$baseline seed=$seed" | tee -a "$LOG"
      else
        code=$?
        echo "[FAILED] task=$task baseline=$baseline seed=$seed exit=$code" | tee -a "$LOG"
        FAILED+=("$task $baseline seed=$seed exit=$code")
      fi
    done
  done
done
python scripts/verify_multidata_batch.py --output-root "$PERTURBED_DATA_ROOT"
echo "[LOG_DIR] $LOG_DIR"
echo "[OUTPUT_ROOT] $NEW_OUTPUT_ROOT"
echo "[FAILED COUNT] ${#FAILED[@]}"
if ((${#FAILED[@]} > 0)); then printf '[FAILED] %s\n' "${FAILED[@]}"; exit 1; fi
