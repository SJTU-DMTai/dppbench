# FlowPipe

This directory adapts [KunyuNi/FlowPipe](https://github.com/KunyuNi/FlowPipe)
at official commit `265a4e91af6eaaf0a734619cb71e183c0788334f` to DPPBench.

## Preserved method core

- One amortized conditional GFlowNet is trained offline across source tasks.
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

The official repository does not publish a ready-to-use checkpoint. FlowPipe
therefore refuses to run with randomly initialized weights. Set
`checkpoint_path` in `config.yaml` to a checkpoint produced by offline,
cross-task training with `FlowPipeTrainer`.

For each target task, cache the official 4096-dimensional Llama embedding at:

```text
<context_cache_dir>/Qwen3-8B/contextual_semantic/<task>.pt
```

`llama_model_path` is an explicit `FlowPipe` constructor argument and defaults
to `/mnt/sdb/dengjiale/dppbench/llm_ckpt`. It is used only by the offline cache
command; normal `FlowPipe.run()` loads the saved tensor and never loads Llama.
The DPPBench experiment scripts explicitly override it with
`/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B` and label the cached context as
`Qwen3-8B`.

Precompute embeddings before training or evaluation:

```bash
python scripts/cache_flowpipe_context.py \
  --data_names default_credit \
  --data_dir /mnt/sdb/dengjiale/dppbench/data \
  --output_dir /mnt/sdb/dengjiale/dppbench/output \
  --llama_model_path /mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B \
  --gpu_id 0
```

Target-task training is intentionally not performed by `FlowPipe.run()`.

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
