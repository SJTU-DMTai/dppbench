#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXP_NAME="exp1_2_marginal_return"
EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
BUDGET_LEVELS="${BUDGET_LEVELS:-2 4 6 8 10}"

# Current evaluate_baseline.py exposes max pipeline length as the common budget
# knob. A future --max_evals knob can be added here without changing callers.
for budget in ${BUDGET_LEVELS}; do
  for seed in ${SEEDS}; do
    for baseline in ${BASELINES}; do
      run_eval \
        "${baseline}" \
        "${seed}" \
        "${EXP_DIR}/budget_${budget}/seed_${seed}" \
        "${budget}" \
        "${EXP_DIR}/budget_${budget}/seed_${seed}/${baseline}/results.csv"
    done
  done
done

collect_csv_columns \
  "${EXP_DIR}" \
  "${EXP_DIR}/analysis/marginal_return_input.csv" \
  "task,baseline,model,max_pipeline_len,std_test_metric_name,std_test_metric,std_test_auc,construct_time_s,eval_time_s,n_steps,ops,n_unique_evals,baseline_is_legal,error,output_dir"
