#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXP_NAME="exp3_1_generation_legality"
EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
SOURCE_DIR="${ANALYSIS_SOURCE_DIR:-${OUTPUT_ROOT}/exp1_1_cost_effectiveness}"

require_exp1_source "${SOURCE_DIR}"

collect_csv_columns \
  "${SOURCE_DIR}" \
  "${EXP_DIR}/analysis/generation_legality_input.csv" \
  "task,baseline,model,baseline_is_legal,baseline_eval_error,std_test_error,error,n_steps,ops,n_unique_evals,output_dir"
