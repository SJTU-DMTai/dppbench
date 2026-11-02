#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/dengjiale/miniconda3/envs/dppbench/bin/python"
DATA_DIR="/mnt/sdb/dengjiale/dppbench/data"
OUTPUT_DIR="/mnt/sdb/dengjiale/dppbench/output"
GPU_ID="${1:-1}"

for task in amazon_beauty kuairec movielens yelp; do
  echo "[run] ${task} gpu=${GPU_ID}"
  "${PYTHON_BIN}" scripts/train_task_model.py \
    --data_name "${task}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --gpu_id "${GPU_ID}"
done
