#!/usr/bin/env bash
# Reuse the ORIGINAL exp1 entry point. Only data/output/log roots differ.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PERTURBED_DATA_ROOT="${PERTURBED_DATA_ROOT:-/mnt/sdb/jiaqinyi/experiments/default_credit_B_original_v1}"
SEED_LIST="${SEED_LIST:-42 43 44}"
BASELINE_LIST="${BASELINE_LIST:-Random SAGA AlphaClean Learn2Clean AutoPrep DAGPipe DiffPrep}"
HARD_MEMORY_GIB="${HARD_MEMORY_GIB:-85}"
# Match common.sh; override to the same device as the corresponding original run.
GPU_ID="${GPU_ID:-1}"
BATCH="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="${LOG_DIR:-logs/default_credit_B_original_${BATCH}}"
NEW_OUTPUT_ROOT="${NEW_OUTPUT_ROOT:-${PERTURBED_DATA_ROOT}/runs/${BATCH}}"
if [[ -e "$NEW_OUTPUT_ROOT" ]]; then
  echo "[ERROR] Refusing existing output root: $NEW_OUTPUT_ROOT" >&2
  exit 2
fi
python scripts/prepare_default_credit_b_original_protocol.py --output-root "$PERTURBED_DATA_ROOT" --verify
mkdir -p "$LOG_DIR"
read -r -a seeds <<< "$SEED_LIST"
read -r -a baselines <<< "$BASELINE_LIST"
FAILED=()
for seed in "${seeds[@]}"; do
  mkdir -p "${LOG_DIR}/seed${seed}"
  for baseline in "${baselines[@]}"; do
    LOG="${LOG_DIR}/seed${seed}/default_credit_${baseline}_seed${seed}.log"
    if [[ -e "$LOG" ]]; then echo "[ERROR] Existing log: $LOG" >&2; exit 2; fi
    echo "[START] baseline=$baseline seed=$seed log=$LOG"
    if TASKS=default_credit SEEDS="$seed" BASELINES="$baseline" \
       DATA_DIR="$PERTURBED_DATA_ROOT" OUTPUT_ROOT="$NEW_OUTPUT_ROOT" GPU_ID="$GPU_ID" \
       python scripts/run_memory_limited.py --memory-gib "$HARD_MEMORY_GIB" -- \
       bash scripts/exps/01_exp1_1_cost_effectiveness.sh > "$LOG" 2>&1; then
      echo "[SUCCESS] baseline=$baseline seed=$seed" | tee -a "$LOG"
    else
      code=$?
      echo "[FAILED] baseline=$baseline seed=$seed exit=$code" | tee -a "$LOG"
      FAILED+=("$baseline seed=$seed exit=$code")
    fi
  done
done
python scripts/prepare_default_credit_b_original_protocol.py --output-root "$PERTURBED_DATA_ROOT" --verify
echo "[LOG_DIR] $LOG_DIR"
echo "[OUTPUT_ROOT] $NEW_OUTPUT_ROOT"
echo "[FAILED COUNT] ${#FAILED[@]}"
if ((${#FAILED[@]} > 0)); then printf '[FAILED] %s\n' "${FAILED[@]}"; exit 1; fi
