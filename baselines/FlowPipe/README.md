# FlowPipe

This directory adapts [KunyuNi/FlowPipe](https://github.com/KunyuNi/FlowPipe)
at official commit `265a4e91af6eaaf0a734619cb71e183c0788334f` to DPPBench.

## Preserved method core

- The DPPBench adaptation trains a separate conditional GFlowNet on each task.
- A trajectory first selects a logical component order and then one primitive
  (or No-Op) for each component.
- The policy uses the official state blocks: logic flag, logic one-hot, stage
  one-hot, downstream-model id, selected-action multi-hot, and 100-column
  statistical features.
- Frozen Llama-3.1-8B contextual-semantic embeddings condition two FiLM layers
  and a context-dependent `logZ` head.
- Training uses replay, reward temperature, failure weighting, and the exact
  executable official TB loss convention.
- Inference samples three trajectories by default and returns the candidate
  with the highest validation reward.

## Required DPPBench adaptations

- The private six-component sklearn catalog is replaced by the complete shared
  DPPBench catalog. Its four benchmark categories define the logical templates.
  Global action ids remain stable across task types; inapplicable actions are
  masked.
- Candidate execution and terminal validation use the shared evaluator.
- The exact generated pipeline receives DAG metadata and must pass the shared
  `check_pipeline()` contract. Operators are never silently inserted or
  rewritten.
- Frozen standard-test rows remain unavailable during synthesis and are replayed
  only by `scripts/evaluate_baseline.py`.

## Checkpoint and context requirements

The DPPBench adapter now trains a fresh policy separately for each target task.
No cross-task pretrained checkpoint is required or loaded. This intentionally
differs from the original offline/amortized method. Training trajectories and
evaluations count against the same task search budget; standard-test metrics
are not used as policy rewards. The task-specific weights are saved as `gfn.pt`.
`train_epochs`, `trajectories_per_epoch`, `updates_per_epoch`, and
`trajectory_batch_size` control training. Validation uses 2 epochs with 2
trajectories each and records `training_stats` / `policy_updates`.

For each target task, cache the 4096-dimensional Qwen3-8B embedding at:

```text
<context_cache_dir>/Qwen3-8B/contextual_semantic/<task>.pt
```

`llama_model_path` is an explicit `FlowPipe` constructor argument and defaults
to `/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B`. It is used only by the
offline cache command; normal `FlowPipe.run()` loads the saved tensor and never
loads Qwen3-8B.

Precompute embeddings before training or evaluation:

```bash
python baselines/FlowPipe/cache_flowpipe_context.py \
  --data_names default_credit \
  --data_dir /mnt/sdb/dengjiale/dppbench/data \
  --output_dir /mnt/sdb/dengjiale/dppbench/output \
  --llama_model_path /mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B \
  --gpu_id 0
```

Target-task training is performed by `FlowPipe.run()` before inference.

## Official defaults

| Setting | Value |
|---|---:|
| hidden dimension | 512 |
| learning rate | 5e-4 |
| replay capacity | 80000 |
| trajectories per combo | 8 |
| trajectory batch size | 16 |
| failure weight | 0.2 |
| reward temperature | 2.0 |
| inference samples | 3 |
