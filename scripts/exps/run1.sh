cd /home/jiaqinyi/code/dppbench

mkdir -p logs/exp_1_1

GPU_ID=1 bash <<'BASH' > logs/exp_1_1/exp1_1_dagpipe_credit_42.log 2>&1
set -euo pipefail

export BASELINES="DAGPipe"

export SEEDS="42"

DATASETS=(
 amazon_beauty 
 kuairec 
 movielens 
 yelp 
 elliptic_bitcoin
  beijing_air_quality 
  berka bike_sharing 
  citibike_jc_hourly 
  default_credit 
  fraud_detection 
  home_credit 
  nyc_taxi_hourly
  polish_bankruptcy
)

for task in "${DATASETS[@]}"
do
  echo "======================================"
  echo "[START] task=$task time=$(date)"
  echo "[BASELINES] $BASELINES"
  echo "======================================"

  TASKS="$task" \
  OUTPUT_ROOT="/mnt/sdb/jiaqinyi/dppbench/output/exps_1_1/$task" \
  GPU_ID=1 \
    bash scripts/exps/01_exp1_1_cost_effectiveness.sh

  echo "[DONE] task=$task time=$(date)"
done

echo "[ALL DONE] $(date)"
BASH