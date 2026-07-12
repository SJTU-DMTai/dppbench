#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXP_NAME="exp2_3_data_model_scale"
EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
SOURCE_DIR="${ANALYSIS_SOURCE_DIR:-${OUTPUT_ROOT}/exp1_1_cost_effectiveness}"

require_exp1_source "${SOURCE_DIR}"

collect_csv_columns \
  "${SOURCE_DIR}" \
  "${EXP_DIR}/analysis/data_model_scale_input.csv" \
  "task,baseline,model,std_test_n_rows,std_test_metric_name,std_test_metric,std_test_auc,std_test_inference_seconds,final_model_training_seconds,final_model_inference_seconds,construct_time_s,eval_time_s,n_steps,n_unique_evals,error,output_dir"
