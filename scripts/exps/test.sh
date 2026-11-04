set -euo pipefail

root=outputs/elliptic_ref16_vs_256
for seed in 42 44 ; do
  for arm in ref16 ref256; do
    checkpoint="$root/seed${seed}/$arm/best_pipeline.yaml"
    test -s "$checkpoint" || { echo "Missing: $checkpoint" >&2; exit 1; }

    out="$root/final_hpo/seed${seed}/$arm"
    mkdir -p "$out"
    python -u scripts/evaluate_baseline.py \
      --baseline DAGPipe \
      --data_names elliptic_bitcoin \
      --model GraphSAGE \
      --data_dir /mnt/sdb/dengjiale/dppbench/data \
      --resume_pipeline "$checkpoint" \
      --final_hpo \
      --seed "$seed" \
      --gpu_id 2 \
      --max_pipeline_len 10 \
      --output_dir "$out" \
      --output_csv "$out/results.csv" \
      2>&1 | tee "$out/run.log"
  done
done