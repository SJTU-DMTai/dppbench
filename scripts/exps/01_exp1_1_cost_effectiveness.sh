#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXP_NAME="exp1_1_cost_effectiveness"
EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"

for seed in ${SEEDS}; do
  for baseline in ${BASELINES}; do
    run_eval \
      "${baseline}" \
      "${seed}" \
      "${EXP_DIR}/seed_${seed}" \
      "${MAX_PIPELINE_LEN}" \
      "${EXP_DIR}/seed_${seed}/${baseline}/results.csv"
  done
done

collect_csv_columns \
  "${EXP_DIR}" \
  "${EXP_DIR}/analysis/cost_effectiveness_input.csv" \
  "task,baseline,model,max_pipeline_len,std_test_metric_name,std_test_metric,std_test_auc,std_test_inference_seconds,baseline_time_kind,baseline_search_inference_seconds,baseline_training_seconds,final_model_training_seconds,final_model_inference_seconds,final_hpo_seconds,final_hpo_trials,final_hpo_best_params,final_eval_seconds,construct_time_s,eval_time_s,n_steps,ops,n_unique_evals,baseline_is_legal,baseline_eval_error,std_test_error,error,output_dir"
