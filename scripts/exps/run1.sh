#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# 加载公共配置
source scripts/exps/common.sh

# 实验配置
SEED_LIST="42"

TASK_LIST=(
  amazon_beauty
  kuairec
  movielens
  elliptic_bitcoin
  beijing_air_quality
  berka
  bike_sharing
  default_credit
  fraud_detection
  home_credit
  polish_bankruptcy
)

BASELINES=(
  SAGA 
  AutoPrep 
  Learn2Clean 
  AlphaClean
  Random
  DiffPrep
  DAGPipe
)

# TASK_LIST=(
#   fraud_detection
# )

HARD_MEMORY_GIB=85

LOG_DIR="logs/exp_1_1/seed42"
mkdir -p "$LOG_DIR"

FAILED=()

for seed in $SEED_LIST; do
  for task in "${TASK_LIST[@]}"; do
    for baseline in "${BASELINES[@]}"; do

      # 每个实验单独保存日志
      LOG="${LOG_DIR}/${task}_${baseline}_seed${seed}.log"

      echo "=================================================="
      echo "[START] task=$task baseline=$baseline seed=$seed"
      echo "[LOG] $LOG"
      echo "[TIME] $(date)"

      # 每次执行都会启动新的 Python 进程
      if TASKS="$task" \
         SEEDS="$seed" \
         BASELINES="$baseline" \
         DPPBENCH_YELP_SIDE_TABLE_SCOPE="full" \
         python scripts/run_memory_limited.py \
           --memory-gib "$HARD_MEMORY_GIB" -- \
           bash scripts/exps/01_exp1_1_cost_effectiveness.sh \
           > "$LOG" 2>&1
      then
        echo "[SUCCESS] task=$task baseline=$baseline seed=$seed"
      else
        code=$?
        echo "[FAILED] task=$task baseline=$baseline seed=$seed exit=$code"
        FAILED+=("$task $baseline seed=$seed exit=$code")
      fi

    done
  done
done

echo "=================================================="
echo "[ALL DONE] $(date)"
echo "[FAILED COUNT] ${#FAILED[@]}"

if ((${#FAILED[@]} > 0)); then
  printf '[FAILED] %s\n' "${FAILED[@]}"
  exit 1
fi