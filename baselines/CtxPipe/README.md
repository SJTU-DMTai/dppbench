# CtxPipe baseline

This directory adapts the original CtxPipe algorithm to DPPBench's shared
operator, DAG, legality, and downstream-evaluation interfaces. The learning
algorithm follows upstream CtxPipe; the deliberate structural difference is
that the merged DPPBench operator space has four categories, so there are four
component `RnnDQN` policies instead of upstream's five learned component
policies. The logical-pipeline `DQN` remains separate.

## Alignment

| Aspect | DPPBench CtxPipe |
|---|---|
| Component policies | Four independent `RnnDQN` models, one per merged operator category |
| Logical-order policy | One `DQN` selecting one of the 24 permutations of four categories |
| Context | Local Qwen3-8B, 100-row CSV sample, 512-token truncation, attention-mask average pooling, 4096 dimensions |
| State | 100 columns x 33 statistics, four primitive-sequence slots, predictor id, logical-pipeline id |
| Networks | Upstream MLP/LSTM dimensions and LeakyReLU/Tanh activations |
| Exploration | Exponential epsilon schedule from 1.0 to 0.2 with decay 2000 |
| Replay/training | Shared FIFO replay, OCG closed/gated/open passes, no target network, Adam at `1e-5` |
| Reward | Zero for intermediate steps and shared downstream-model fitness at termination |

The default encoder is loaded locally from:

```text
/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B
```

The encoder deterministically samples up to 100 rows and caches the 4096-value
embedding by the sampled CSV content hash under:

```text
/mnt/sdb/dengjiale/dppbench/outputs/CtxPipe/context_cache/<task>/
```

This keeps different tasks and transformed table states separate. Context is
requested once for replay storage and again on a greedy action, matching the
upstream action path; repeated requests for identical table content are served
from memory or disk without running Qwen3-8B again. Random actions do not make
the second request.

## DPPBench adaptation boundary

CtxPipe chooses the logical category order and one action from each category.
DPPBench's shared layer supplies the available operators, materializes operator
parameters and DAG edges, checks partial/final legality, runs preprocessing, and
computes the downstream reward. `SKIP` is included in every category action
space so the shared mandatory-operator repair and final legality checks remain
authoritative.

Search fitness is validation-only. Frozen `std_test` replay remains in the
benchmark harness and is not used as an RL reward.

Local weight loading uses `DPPBENCH_LLM_QUANTIZATION=auto` by default: FP8
when CUDA hardware and torchao support it, otherwise unquantized weights with
a warning. Set `none` to disable quantization or `fp8` to require it strictly.
CPU loading uses FP32; this does not imply an 8B model fits available memory.
Quantization is optional; install compatible torchao/transformers on the GPU
server if required. Model choices and downstream-feedback defaults are retained.
