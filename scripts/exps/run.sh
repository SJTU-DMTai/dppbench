cd /home/jiaqinyi/code/dppbench

mkdir -p logs/exp_1_1

bash <<'BASH' > logs/exp_1_1/exp1_cost_yelp_alpha_43_44.log 2>&1
set -euo pipefail

FAILED_TASKS=()
SUCCESS_TASKS=()

SEED_LIST="43 44"

for task in yelp
do
  echo "============================================================"
  echo "[START] task=$task seeds=$SEED_LIST time=$(date)"
  echo "============================================================"

  if TASKS="$task" \
     OUTPUT_ROOT="/mnt/sdb/jiaqinyi/dppbench/output/exps_1_1/$task" \
     SEEDS="$SEED_LIST" \
     bash scripts/exps/01_exp1_1_cost_effectiveness.sh
  then
    echo "[SUCCESS] task=$task seeds=$SEED_LIST time=$(date)"
    SUCCESS_TASKS+=("$task")
  else
    exit_code=$?

    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "[FAILED] task=$task seeds=$SEED_LIST exit_code=$exit_code"
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