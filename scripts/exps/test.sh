mkdir -p logs/yelp_acceptance_43

export DPPBENCH_YELP_SIDE_TABLE_SCOPE=full
export DPPBENCH_MEMORY_TRACE=1

python scripts/run_memory_limited.py --memory-gib 75 -- \
  python -u scripts/evaluate_baseline.py \
    --baseline AlphaClean \
    --data_names yelp \
    --model DIN \
    --seed 43 \
    --gpu_id 1 \
    --data_dir /mnt/sdb/dengjiale/dppbench/data \
    --output_dir /mnt/sdb/jiaqinyi/dppbench/output/yelp_acceptance_43 \
    --output_csv /mnt/sdb/jiaqinyi/dppbench/output/yelp_acceptance_43/results.csv \
    --max_pipeline_len 10 \
    --max_evaluations 20 \
    --max_memory_mb 71680 \
    --final_hpo \
    > logs/yelp_acceptance_43/run.log 2>&1

echo "exit_code=$?" >> logs/yelp_acceptance_43/run1.log