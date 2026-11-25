SOURCE_ROOT="/mnt/sdb/jiaqinyi/dppbench/output/exps1_10_30/exp1_1_cost_effectiveness"
TRANSFER_ROOT="/mnt/sdb/jiaqinyi/dppbench/output/exps1_10_30/model_transfer_all_home"
DATA_ROOT="/mnt/sdb/dengjiale/dppbench/data"

TASKS="home_credit"
BASELINES="Random,AlphaClean,SAGA,AutoPrep,Learn2Clean,DiffPrep,DAGPipe"

for seed in 42 43 44; do
  python scripts/evaluate_model_transfer.py \
    --source_root "$SOURCE_ROOT" \
    --pipeline_template '{source_root}/seed_{seed}/{baseline}/{task}_{source_model}/best_pipeline.yaml' \
    --data_names "$TASKS" \
    --baselines "$BASELINES" \
    --models all \
    --seed "$seed" \
    --data_dir "$DATA_ROOT" \
    --output_dir "$TRANSFER_ROOT" \
    --gpu_id 1
done