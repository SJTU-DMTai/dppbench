#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXP_NAME="exp4_3_search_space_coverage"
EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
SOURCE_DIR="${ANALYSIS_SOURCE_DIR:-${OUTPUT_ROOT}/exp1_1_cost_effectiveness}"

require_exp1_source "${SOURCE_DIR}"

collect_csv_columns \
  "${SOURCE_DIR}" \
  "${EXP_DIR}/analysis/search_space_coverage_input.csv" \
  "task,baseline,model,n_unique_evals,ops,n_steps,std_test_metric_name,std_test_metric,std_test_auc,baseline_eval_error,error,output_dir"
