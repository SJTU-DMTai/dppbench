cd /home/jiaqinyi/code/dppbench

mkdir -p logs/exp_1_1

bash <<'BASH' > logs/exp_1_1/exp1_cost_nollm_43.log 2>&1
set -euo pipefail

FAILED_TASKS=()
SUCCESS_TASKS=()

for task in \
  amazon_beauty kuairec movielens yelp elliptic_bitcoin \
  beijing_air_quality berka bike_sharing citibike_jc_hourly \
  default_credit fraud_detection home_credit nyc_taxi_hourly \
  polish_bankruptcy
do
  echo "============================================================"
  echo "[START] task=$task seed=43 time=$(date)"
  echo "============================================================"

  if TASKS="$task" \
     OUTPUT_ROOT="/mnt/sdb/jiaqinyi/dppbench/output/exps_1_1/$task" \
     SEEDS=43 \
     bash scripts/exps/01_exp1_1_cost_effectiveness.sh
  then
    echo "[SUCCESS] task=$task seed=43 time=$(date)"
    SUCCESS_TASKS+=("$task")
  else
    exit_code=$?

    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "[FAILED] task=$task seed=43 exit_code=$exit_code"
    echo "[FAILED] time=$(date)"
    echo "[WARNING] Skipping failed task and continuing..."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    FAILED_TASKS+=("$task")
  fi

  echo
done

echo "============================================================"
echo "[ALL DONE] $(date)"
echo "[SUMMARY] Success: ${#SUCCESS_TASKS[@]}"
echo "[SUMMARY] Failed: ${#FAILED_TASKS[@]}"

if ((${#FAILED_TASKS[@]} > 0)); then
  printf '[FAILED TASK] %s\n' "${FAILED_TASKS[@]}"
fi

echo "============================================================"
BASH