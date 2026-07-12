#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXP_NAME="exp1_3_transferability"
EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
SOURCE_DIR="${ANALYSIS_SOURCE_DIR:-${OUTPUT_ROOT}/exp1_1_cost_effectiveness}"

require_exp1_source "${SOURCE_DIR}"

collect_csv_columns \
  "${SOURCE_DIR}" \
  "${EXP_DIR}/analysis/transferability_input.csv" \
  "task,baseline,model,std_test_metric_name,std_test_metric,std_test_auc,baseline_time_kind,baseline_training_seconds,baseline_search_inference_seconds,construct_time_s,n_steps,ops,error,output_dir"
