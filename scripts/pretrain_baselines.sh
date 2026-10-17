#!/usr/bin/env bash
# Run from anywhere. The agent weights are grouped by baseline/downstream model.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-/mnt/sdb/dengjiale/dppbench/data}"
PRETRAINED_DIR="${PRETRAINED_DIR:-$PWD/checkpoints/pretrained}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
SMALL_N="${SMALL_N:-1500}"
BASELINES="${BASELINES:-HAIPipe CtxPipe FlowPipe}"
# Both CtxPipe and FlowPipe use this frozen 8B encoder in this adaptation.
# Download once if needed (huggingface_hub CLI):
#   hf download Qwen/Qwen3-8B --local-dir /path/to/Qwen3-8B
QWEN3_MODEL_PATH="${QWEN3_MODEL_PATH:-/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B}"
export DPPBENCH_LLM_QUANTIZATION="${DPPBENCH_LLM_QUANTIZATION:-auto}"

# Defaults pretrain on all shipped, supported tabular/recommendation tasks,
# including later targets: this is the seen-task training setup discussed in the README.
# FlowPipe graph support is not implemented. Tenrec is not shipped.
TABULAR_TASKS="${TABULAR_TASKS:-berka,default_credit,fraud_detection,home_credit,polish_bankruptcy,beijing_air_quality,bike_sharing,citibike_jc_hourly,nyc_taxi_hourly}"
REC_TASKS="${REC_TASKS:-amazon_beauty,kuairec,movielens,yelp}"
TABULAR_MODEL="${TABULAR_MODEL:-LightGBM}"
REC_MODEL="${REC_MODEL:-DIN}"
GROUPS_TO_RUN="${GROUPS_TO_RUN:-tabular rec}"
FRAMES="${FRAMES:-56000}"
FLOW_EPOCHS="${FLOW_EPOCHS:-20}"
HELD_OUT_TASKS="${HELD_OUT_TASKS:-}"
BUILD_FROZEN="${BUILD_FROZEN:-1}"

if [[ "$BUILD_FROZEN" == 1 ]]; then
  for group in $GROUPS_TO_RUN; do
    case "$group" in
      tabular) tasks="$TABULAR_TASKS" ;;
      rec) tasks="$REC_TASKS" ;;
      *) echo "Unknown group: $group" >&2; exit 1 ;;
    esac
    "$PYTHON" scripts/build_std_test.py --data_names "$tasks" --data_dir "$DATA_DIR" --gpu_id -1
  done
fi

for baseline in $BASELINES; do
  if [[ "$baseline" != HAIPipe && ! -f "$QWEN3_MODEL_PATH/config.json" ]]; then
    echo "Missing Qwen3-8B at $QWEN3_MODEL_PATH; set QWEN3_MODEL_PATH before training." >&2
    exit 1
  fi
  for group in $GROUPS_TO_RUN; do
    case "$group" in
      tabular) tasks="$TABULAR_TASKS"; model="$TABULAR_MODEL" ;;
      rec) tasks="$REC_TASKS"; model="$REC_MODEL" ;;
      *) echo "Unknown group: $group" >&2; exit 1 ;;
    esac
    "$PYTHON" scripts/pretrain_baselines.py \
      --baseline "$baseline" --data_names "$tasks" --model "$model" \
      --data_dir "$DATA_DIR" --pretrained_dir "$PRETRAINED_DIR" \
      --held_out_data_names "$HELD_OUT_TASKS" \
      --gpu_id "$GPU_ID" --seed "$SEED" --small_n "$SMALL_N" \
      --max_pipeline_len 10 --frames "$FRAMES" --epochs "$FLOW_EPOCHS" \
      --llm_model_path "$QWEN3_MODEL_PATH" "$@"
  done
done
