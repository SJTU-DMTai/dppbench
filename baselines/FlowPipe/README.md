# FlowPipe

This directory adapts [KunyuNi/FlowPipe](https://github.com/KunyuNi/FlowPipe)
at official commit `265a4e91af6eaaf0a734619cb71e183c0788334f` to DPPBench.

## Preserved method core

- A shared conditional GFlowNet supports scratch, frozen-pretrained and pretrained-finetune modes.
- A trajectory first selects a logical component order and then one primitive
  (or No-Op) for each component.
- The policy uses the official state blocks: logic flag, logic one-hot, stage
  one-hot, downstream-model id, selected-action multi-hot, and 100-column
  statistical features.
- Frozen Qwen3-8B contextual-semantic embeddings condition two FiLM layers
  and a context-dependent `logZ` head.
- Training uses replay, reward temperature, failure weighting, and the exact
  executable official TB loss convention.
- Inference samples three trajectories by default and returns the candidate
  with the highest raw validation fitness.

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

Default deployment loads a shared source-trained policy and fine-tunes it on the target
(`training_mode: finetune`). `scratch` retains random-initialization task training;
`pretrained` loads the same checkpoint and skips all policy updates. Each target
starts from an independent copy. Missing/incompatible weights fail explicitly.
See [pretraining and evaluation commands](../../scripts/BASELINE_TRAINING.md).
The final candidates come from post-training inference, not source training traces.

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

Target-task training is performed only in `scratch` or `finetune`; `pretrained` freezes the policy.

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

## Regression metrics and loading

The evaluator chooses the metric from the task config. For RMSE/MAE/MSE/logloss,
fitness is the negative error and the positive training reward is `1/(1+error)`
(scale 1). AUC keeps its value. Raw metrics and fitness are retained in logs;
the final pipeline is ranked by fitness, not the transformed reward.

Generic unlabelled embedding files are no longer reused across encoders.
Use the model-specific cache path above and regenerate it when changing the
checkpoint, quantization, data, or seed.

Local weight loading uses `DPPBENCH_LLM_QUANTIZATION=auto` by default: FP8
when CUDA hardware and torchao support it, otherwise unquantized weights with
a warning. Set `none` to disable quantization or `fp8` to require it strictly.
CPU loading uses FP32; this does not imply an 8B model fits available memory.
Quantization is optional; install compatible torchao/transformers on the GPU
server if required. Model choices and downstream-feedback defaults are retained.
