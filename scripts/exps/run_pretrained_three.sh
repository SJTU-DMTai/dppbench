#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/exps/common.sh

# Override common.sh defaults for this checkpoint handoff.
DATA_DIR=/mnt/sdb/dengjiale/dppbench/data
DELIVERY=/mnt/sdb/jiaqinyi/ckpts/pretrained_delivery_20261102_relocated
QWEN3=/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B
OUTPUT_ROOT=/mnt/sdb/jiaqinyi/dppbench/output/pretrained_three
PYTHON="$(command -v python)"
SEED_LIST="43 44"
GPU_ID=1
HARD_MEMORY_GIB=85
TRAINING_MODE=finetune
FINETUNE_FRAMES=2000
FINETUNE_EPOCHS=2

# The 13 tasks covered by the delivered FlowPipe caches and checkpoints.
TASK_LIST=(
  amazon_beauty
  kuairec
  movielens 
  beijing_air_quality berka bike_sharing citibike_jc_hourly
  default_credit fraud_detection home_credit nyc_taxi_hourly polish_bankruptcy
)
BASELINES=(HAIPipe)
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="$OUTPUT_ROOT/$RUN_ID"
FAILED=()
export DPPBENCH_LLM_QUANTIZATION=none
export DPPBENCH_YELP_SIDE_TABLE_SCOPE=full
# --gpu_id is a physical ID. Avoid inherited CUDA filtering remapping it.
unset CUDA_VISIBLE_DEVICES

model_for_task() {
  case "$1" in
    amazon_beauty|kuairec|movielens|yelp) echo DIN ;;
    *) echo LightGBM ;;
  esac
}

# Fail before the batch if required data, checkpoints or caches are missing.
for task in "${TASK_LIST[@]}"; do
  model="$(model_for_task "$task")"
  frozen=train_frozen.parquet
  [[ "$model" != DIN ]] || frozen=interaction_frozen.parquet
  for name in meta.json std_test.parquet "$frozen"; do
    [[ -f "$DATA_DIR/$task/std_test/$name" ]] || {
      echo "[ERROR] Missing $DATA_DIR/$task/std_test/$name" >&2; exit 1;
    }
  done
  [[ -f "$DELIVERY/FlowPipe/$model/context_cache/Qwen3-8B/contextual_semantic/$task.pt" ]] || {
    echo "[ERROR] Missing FlowPipe cache: $task / $model" >&2; exit 1;
  }
done
for baseline in "${BASELINES[@]}"; do
  models=(LightGBM DIN)
  [[ "$baseline" != HAIPipe ]] || models=(LightGBM)
  for model in "${models[@]}"; do
    files=(pretrain.json checkpoint.pt)
    [[ "$baseline" != HAIPipe ]] || files=(pretrain.json aipipe.pt alternative_evaluator.pt)
    for name in "${files[@]}"; do
      [[ -s "$DELIVERY/$baseline/$model/$name" ]] || {
        echo "[ERROR] Missing $DELIVERY/$baseline/$model/$name" >&2; exit 1;
      }
    done
  done
done
[[ -f "$QWEN3/config.json" ]] || { echo "[ERROR] Missing Qwen3 config" >&2; exit 1; }
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
# This requires user systemd memory-controller delegation, not sudo.
# Stop immediately if the hard memory limit cannot be enforced.
"$PYTHON" scripts/run_memory_limited.py --memory-gib "$HARD_MEMORY_GIB" -- "$PYTHON" -c 'print("Memory scope OK")'
mkdir -p "$RUN_ROOT"
cp "${BASH_SOURCE[0]}" "$RUN_ROOT/launcher.sh"

for seed in $SEED_LIST; do
  OUT="$RUN_ROOT/seed$seed"
  LOG_DIR="logs/exp1_11_20_llm/seed${seed}"
  mkdir -p "$LOG_DIR"
  for task in "${TASK_LIST[@]}"; do
    model="$(model_for_task "$task")"
    for baseline in "${BASELINES[@]}"; do
      FOLDER="$OUT/$baseline/${task}_${model}"
      LOG="$LOG_DIR/${task}_${baseline}_seed${seed}.log"
      EXTRA=()
      case "$baseline" in
        FlowPipe) EXTRA=(--qwen3_model_path "$QWEN3") ;;
        CtxPipe) EXTRA=(--ctxpipe_gte_model_path "$QWEN3"
          --context_cache_dir "$OUT/context_cache/CtxPipe/$task") ;;
      esac
      CMD=("$PYTHON" -u scripts/evaluate_baseline.py
        --baseline "$baseline" --data_names "$task" --model "$model"
        --training_mode "$TRAINING_MODE" --pretrained_dir "$DELIVERY"
        --finetune_frames "$FINETUNE_FRAMES" --finetune_epochs "$FINETUNE_EPOCHS"
        --data_dir "$DATA_DIR" --gpu_id "$GPU_ID" --seed "$seed"
        --small_n 30000 --fast_train --skip_build
        --output_dir "$OUT" --output_csv "$FOLDER/metrics.csv"
        "${EXTRA[@]}")
      echo "[START] $(date) task=$task baseline=$baseline model=$model seed=$seed"
      echo "[LOG] $LOG"
      mkdir -p "$FOLDER"
      printf '%q ' "${CMD[@]}" > "$FOLDER/command.sh"
      printf '\n' >> "$FOLDER/command.sh"
      if "$PYTHON" scripts/run_memory_limited.py --memory-gib "$HARD_MEMORY_GIB" -- "${CMD[@]}" > "$LOG" 2>&1; then
        if "$PYTHON" -c 'from pathlib import Path; import sys; from scripts.run_full_baseline_experiments import validate_result; validate_result(Path(sys.argv[1]))' "$FOLDER" >> "$LOG" 2>&1; then
          echo "[SUCCESS] $task $baseline seed=$seed"
          printf '{"success": true}\n' > "$FOLDER/status.json"
        else
          echo "[FAILED] Result validation: $task $baseline seed=$seed"
          FAILED+=("$task $baseline seed=$seed result-validation")
          printf '{"success": false, "error": "result validation failed; see log"}\n' > "$FOLDER/status.json"
        fi
      else
        code=$?
        echo "[FAILED] $task $baseline seed=$seed exit=$code"
        FAILED+=("$task $baseline seed=$seed exit=$code")
        printf '{"success": false, "exit_code": %s}\n' "$code" > "$FOLDER/status.json"
      fi
    done
  done
done
echo "[ALL DONE] $(date) OUTPUT=$RUN_ROOT"
echo "[FAILED COUNT] ${#FAILED[@]}"
if ((${#FAILED[@]} > 0)); then
  printf '%s\n' "${FAILED[@]}" | tee "$RUN_ROOT/failed.txt"
  exit 1
fi
