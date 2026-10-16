cd /home/jiaqinyi/code/dppbench

mkdir -p logs/exp_1_1

bash <<'BASH' > logs/exp_1_1/exp1_1_llmtest_42.log 2>&1
set -euo pipefail

for task in \
  amazon_beauty kuairec movielens yelp elliptic_bitcoin \
  beijing_air_quality berka bike_sharing citibike_jc_hourly \
  default_credit fraud_detection home_credit nyc_taxi_hourly \
  polish_bankruptcy
do
  echo "[START] task=$task time=$(date)"

  TASKS="$task" \
  OUTPUT_ROOT="/mnt/sdb/jiaqinyi/dppbench/output/exps_1_1/$task" \
    bash scripts/exps/01_exp1_1_cost_effectiveness.sh

  echo "[DONE] task=$task time=$(date)"
done

echo "[ALL DONE] $(date)"
BASH