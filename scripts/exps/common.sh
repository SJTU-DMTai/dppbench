#!/usr/bin/env bash

set -euo pipefail

EXPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXPS_DIR}/../.." && pwd)"

DATA_DIR="${DATA_DIR:-/mnt/sdb/dengjiale/dppbench/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/sdb/jiaqinyi/dppbench/output/exps_1_1}"
GPU_ID="${GPU_ID:-1}"
CTXPIPE_GTE_MODEL_PATH="${CTXPIPE_GTE_MODEL_PATH:-/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B}"
DEEPPREP_MODEL_PATH="${DEEPPREP_MODEL_PATH:-/mnt/sdb/dengjiale/dppbench/llm_ckpt/DeepPrep-Qwen3-8B}"
QWEN3_MODEL_PATH="${QWEN3_MODEL_PATH:-/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B}"
MAX_PIPELINE_LEN="${MAX_PIPELINE_LEN:-10}"
MAX_EVALUATIONS="${MAX_EVALUATIONS:-1000}"
MAX_MEMORY_MB="${MAX_MEMORY_MB:-71680}"
MAX_GPU_MEMORY_MB="${MAX_GPU_MEMORY_MB:-81920}"
MAX_LLM_CALLS="${MAX_LLM_CALLS:-100}"

usage() {
  cat <<EOF
Usage: $0 [--gpu_id GPU_ID] [--cpu] [budget options]

Options:
  --gpu_id GPU_ID   GPU index passed to scripts/evaluate_baseline.py.
                    Use -1 for CPU. Default: ${GPU_ID}
  --gpu_id=GPU_ID   Same as --gpu_id GPU_ID.
  --cpu             Equivalent to --gpu_id -1.
  --max_pipeline_len N
                    Maximum generated pipeline length. Default: ${MAX_PIPELINE_LEN:-10}
  --max_evaluations N
                    Maximum search-time pipeline evaluations. Default: ${MAX_EVALUATIONS}
  --max_llm_calls N Maximum LLM calls. Default: ${MAX_LLM_CALLS}
  --max_memory_mb N Maximum process memory in MB. Default: ${MAX_MEMORY_MB}
  --max_gpu_memory_mb N
                    Maximum GPU memory in MB. Default: ${MAX_GPU_MEMORY_MB}
  -h, --help        Show this help.

Environment variables such as GPU_ID, DATA_DIR, OUTPUT_ROOT, TASKS,
BASELINES, SEEDS, MAX_PIPELINE_LEN, MAX_EVALUATIONS, MAX_LLM_CALLS,
MAX_MEMORY_MB, and MAX_GPU_MEMORY_MB are still supported. Command-line
arguments take precedence.
EOF
}

parse_common_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --max_pipeline_len|--max_evaluations|--max_llm_calls|--max_memory_mb|--max_gpu_memory_mb)
        if [[ $# -lt 2 ]]; then
          echo "[error] $1 requires a value." >&2
          usage >&2
          exit 2
        fi
        case "$1" in
          --max_pipeline_len) MAX_PIPELINE_LEN="$2" ;;
          --max_evaluations) MAX_EVALUATIONS="$2" ;;
          --max_llm_calls) MAX_LLM_CALLS="$2" ;;
          --max_memory_mb) MAX_MEMORY_MB="$2" ;;
          --max_gpu_memory_mb) MAX_GPU_MEMORY_MB="$2" ;;
        esac
        shift 2
        ;;
      --max_pipeline_len=*|--max_evaluations=*|--max_llm_calls=*|--max_memory_mb=*|--max_gpu_memory_mb=*)
        local opt="${1%%=*}"
        local val="${1#*=}"
        case "${opt}" in
          --max_pipeline_len) MAX_PIPELINE_LEN="${val}" ;;
          --max_evaluations) MAX_EVALUATIONS="${val}" ;;
          --max_llm_calls) MAX_LLM_CALLS="${val}" ;;
          --max_memory_mb) MAX_MEMORY_MB="${val}" ;;
          --max_gpu_memory_mb) MAX_GPU_MEMORY_MB="${val}" ;;
        esac
        shift
        ;;
      --gpu_id)
        if [[ $# -lt 2 ]]; then
          echo "[error] --gpu_id requires a value." >&2
          usage >&2
          exit 2
        fi
        GPU_ID="$2"
        shift 2
        ;;
      --gpu_id=*)
        GPU_ID="${1#*=}"
        shift
        ;;
      --cpu)
        GPU_ID="-1"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "[error] unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
}

parse_common_args "$@"

DEFAULT_TASKS="amazon_beauty,kuairec,movielens,yelp,elliptic_bitcoin,beijing_air_quality,berka,bike_sharing,citibike_jc_hourly,default_credit,fraud_detection,home_credit,nyc_taxi_hourly,polish_bankruptcy"
TASKS="${TASKS:-${DEFAULT_TASKS}}"

# DEFAULT_BASELINES="SAGA BAT AutoPrep DeepPrep Learn2Clean AlphaClean Random DiffPrep CtxPipe HAIPipe FlowPipe"

# baselines not based on LLMs
# DEFAULT_BASELINES="DiffPrep"

# baselines based on LLMs
DEFAULT_BASELINES="BAT DeepPrep CtxPipe FlowPipe"

BASELINES="${BASELINES:-${DEFAULT_BASELINES}}"

SEEDS="${SEEDS:-42}"

add_optional_arg() {
  local -n target_cmd="$1"
  local arg_name="$2"
  local arg_value="$3"
  if [[ -n "${arg_value}" ]]; then
    target_cmd+=("${arg_name}" "${arg_value}")
  fi
}

run_eval() {
  local baseline="$1"
  local seed="$2"
  local exp_dir="$3"
  local max_pipeline_len="${4:-}"
  local output_csv="$5"

  mkdir -p "$(dirname "${output_csv}")"

  local cmd=(
    python scripts/evaluate_baseline.py
    --final_hpo
    --baseline "${baseline}"
    --data_names "${TASKS}"
    --data_dir "${DATA_DIR}"
    --output_dir "${exp_dir}"
    --output_csv "${output_csv}"
    --seed "${seed}"
    --gpu_id "${GPU_ID}"
    --ctxpipe_gte_model_path "${CTXPIPE_GTE_MODEL_PATH}"
    --deepprep_model_path "${DEEPPREP_MODEL_PATH}"
    --qwen3_model_path "${QWEN3_MODEL_PATH}"
  )

  case "$baseline" in
    CtxPipe|HAIPipe|FlowPipe)
      cmd+=(--training_mode "${TRAINING_MODE:-finetune}"
            --pretrained_dir "${PRETRAINED_DIR:-${REPO_ROOT}/checkpoints/pretrained}")
      add_optional_arg cmd --finetune_frames "${FINETUNE_FRAMES:-}"
      add_optional_arg cmd --finetune_epochs "${FINETUNE_EPOCHS:-}"
      ;;
  esac

  if [[ -n "${max_pipeline_len}" ]]; then
    cmd+=(--max_pipeline_len "${max_pipeline_len}")
  fi
  add_optional_arg cmd --max_evaluations "${MAX_EVALUATIONS}"
  add_optional_arg cmd --max_memory_mb "${MAX_MEMORY_MB}"
  add_optional_arg cmd --max_gpu_memory_mb "${MAX_GPU_MEMORY_MB}"
  add_optional_arg cmd --max_llm_calls "${MAX_LLM_CALLS}"

  echo "[run] ${cmd[*]}"
  (cd "${REPO_ROOT}" && "${cmd[@]}")
}

collect_csv_columns() {
  local source_dir="$1"
  local output_csv="$2"
  local columns="$3"

  mkdir -p "$(dirname "${output_csv}")"
  python - "$source_dir" "$output_csv" "$columns" <<'PY'
import csv
import os
import sys

source_dir, output_csv, columns_arg = sys.argv[1:4]
columns = [c.strip() for c in columns_arg.split(",") if c.strip()]
fields = ["source_csv"] + columns
rows = []

for root, _, files in os.walk(source_dir):
    for name in files:
        if not name.endswith(".csv"):
            continue
        path = os.path.join(root, name)
        if os.path.realpath(path) == os.path.realpath(output_csv):
            continue
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not {"task", "baseline", "model"}.issubset(reader.fieldnames or []):
                    continue
                # Do not recursively re-aggregate an earlier summary.
                if "source_csv" in reader.fieldnames:
                    continue
                for row in reader:
                    rows.append({"source_csv": path, **{c: row.get(c, "") for c in columns}})
        except Exception as exc:
            rows.append({"source_csv": path, **{c: "" for c in columns}, "error": f"{type(exc).__name__}: {exc}"})

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"[collect] wrote {output_csv} rows={len(rows)}")
PY
}

require_exp1_source() {
  local source_dir="$1"
  if [[ ! -d "${source_dir}" ]]; then
    echo "[error] source directory not found: ${source_dir}" >&2
    echo "[hint] run scripts/exps/01_exp1_1_cost_effectiveness.sh first, or set ANALYSIS_SOURCE_DIR." >&2
    exit 1
  fi
}
