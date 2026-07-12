#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXP_NAME="exp2_2_pipeline_length_scalability"
EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
PIPELINE_LENGTH_LEVELS="${PIPELINE_LENGTH_LEVELS:-2 4 6 8 10}"

for max_len in ${PIPELINE_LENGTH_LEVELS}; do
  for seed in ${SEEDS}; do
    for baseline in ${BASELINES}; do
      run_eval \
        "${baseline}" \
        "${seed}" \
        "${EXP_DIR}/max_len_${max_len}/seed_${seed}" \
        "${max_len}" \
        "${EXP_DIR}/max_len_${max_len}/seed_${seed}/${baseline}/results.csv"
    done
  done
done

collect_csv_columns \
  "${EXP_DIR}" \
  "${EXP_DIR}/analysis/pipeline_length_scalability_input.csv" \
  "task,baseline,model,max_pipeline_len,std_test_metric_name,std_test_metric,std_test_auc,construct_time_s,eval_time_s,n_steps,ops,n_unique_evals,baseline_is_legal,error,output_dir"
