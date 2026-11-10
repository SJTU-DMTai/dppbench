# Pretraining and deployment of HAIPipe, CtxPipe and FlowPipe

All three default to `training_mode: finetune`. They implement:

| Mode | Initialization | Target agent updates |
| --- | --- | --- |
| `scratch` | Random | Existing full task-training budget |
| `pretrained` | Shared pretrained checkpoint | None |
| `finetune` (default) | Independent copy of shared checkpoint | Limited target budget |

HAIPipe always retains its original candidate-level active learning: even in
`pretrained` mode, its pretrained alternative evaluator adapts to target validation
feedback. Only the RL agent is frozen in this mode. Fine-tuning also updates the
RL agent. Standard-test scoring is handled by the separate final evaluator.

## Prepare and pretrain

Activate the project environment and install `requirements.txt`. Both CtxPipe and
FlowPipe in this adaptation use **Qwen3-8B** as a frozen encoder (4096 dimensions).
HAIPipe requires no language model. Obtain the weights once, if not already local:

```bash
hf download Qwen/Qwen3-8B --local-dir /path/to/Qwen3-8B
```

Launch the supplied script from any directory:

```bash
DATA_DIR=/path/to/datasets \
QWEN3_MODEL_PATH=/path/to/Qwen3-8B \
GPU_ID=0 \
bash scripts/pretrain_baselines.sh
```

The script explicitly passes the encoder path, prepares frozen splits, then
trains each baseline sequentially on two groups: nine tabular tasks with
LightGBM and four recommendation tasks with DIN. These are shared agents across
datasets **for each downstream model**, not one agent shared across downstream
models. They include later target tasks: this is *seen-task pretraining*. No
frozen standard-test rows enter the agent profiles or reward execution. Existing
weights are not overwritten unless `--overwrite` is passed.

Useful overrides (environment variables):

- `PYTHON`, `DATA_DIR`, `PRETRAINED_DIR` (default `checkpoints/pretrained`), `GPU_ID`, `SEED`.
- `BASELINES="HAIPipe CtxPipe FlowPipe"`, `GROUPS_TO_RUN="tabular rec"`.
- `TABULAR_TASKS`, `REC_TASKS` (comma-separated), `TABULAR_MODEL`, `REC_MODEL`.
- `FRAMES=56000` (total across source tasks for CtxPipe/HAIPipe, not per task),
  `FLOW_EPOCHS=20`, `SMALL_N=1500`, `BUILD_FROZEN=0` to reuse prepared splits.
- `DPPBENCH_LLM_QUANTIZATION=auto|none|fp8`. Auto uses FP8 only when available;
  fallback to unquantized weights may require more VRAM. The LLM is not fine-tuned.

For example, train only CtxPipe on the tabular group:

```bash
BASELINES=CtxPipe GROUPS_TO_RUN=tabular \
DATA_DIR=/path/to/datasets QWEN3_MODEL_PATH=/path/to/Qwen3-8B GPU_ID=0 \
bash scripts/pretrain_baselines.sh
```

For explicit source/held-out task splits, use the Python CLI:

```bash
python scripts/pretrain_baselines.py --baseline HAIPipe \
  --data_names berka,default_credit,polish_bankruptcy \
  --held_out_data_names fraud_detection,home_credit \
  --model LightGBM --data_dir /path/to/datasets \
  --pretrained_dir checkpoints/pretrained --frames 56000 --gpu_id 0
```

Source and explicitly held-out task sets must be disjoint. Build the frozen
training artifacts first when using the Python CLI directly. Target train/validation
feedback can be used in target adaptation; the frozen standard test cannot.
To compare independent pretraining seeds, use a different `PRETRAINED_DIR` per seed.
FlowPipe graph tasks are unsupported. Tenrec has no shipped loader. For a graph
experiment with CtxPipe/HAIPipe, use the Python CLI with graph sources and a
compatible GNN model; the default launcher does not include this extension.

## Artifacts

```text
checkpoints/pretrained/
  CtxPipe/LightGBM/checkpoint.pt
  HAIPipe/LightGBM/aipipe.pt
  HAIPipe/LightGBM/alternative_evaluator.pt
  FlowPipe/LightGBM/checkpoint.pt
  FlowPipe/LightGBM/context_cache/Qwen3-8B/contextual_semantic/<task>.pt
  .../DIN/...
```

Every model directory contains `pretrain.json`: source tasks, held-out tasks,
seed, source training time, sampling settings, training statistics, action/state
contract and encoder identity. The saved agent is the final source-trained
agent; the script does not perform held-out-task hyperparameter selection.
The provided budgets are starting configurations, not claims of convergence.

CtxPipe switches datasets between complete episodes and shares one frozen 8B
encoder and one agent/replay buffer across source environments. FlowPipe switches
source environment/context combinations and caches the frozen embeddings before
agent training. HAIPipe first trains its shared AIPipe agent, then fits its
alternative evaluator from finite source-candidate scores.

## Test with default fine-tuning

```bash
python scripts/evaluate_baseline.py --baseline CtxPipe \
  --data_names default_credit --model LightGBM --data_dir /path/to/datasets \
  --pretrained_dir checkpoints/pretrained --ctxpipe_model_path /path/to/Qwen3-8B \
  --finetune_frames 2000 --max_pipeline_len 10 --gpu_id 0 \
  --output_dir outputs/agent_finetune
```

Use the same checkpoint root for HAIPipe or FlowPipe. For FlowPipe pass
`--qwen3_model_path /path/to/Qwen3-8B` and optionally `--finetune_epochs 2`.
Set `--training_mode scratch` or `--training_mode pretrained` for the other two groups;
use separate output directories for each mode. Default target budgets are 2000
interaction frames for CtxPipe/HAIPipe and 2 epochs × 8 trajectories for FlowPipe.
The existing shared evaluation/time/resource budget options still apply.

Missing or incompatible checkpoints fail explicitly. Changing the encoder,
operator IDs, downstream model or state contract requires matching retrained
weights. Existing v2 HAIPipe files and old CtxPipe gate-less files do not have the
new contract and are rejected. Source files cannot be overwritten by a target run.
The Qwen path recorded during training must match deployment (or the checkpoint
must be deliberately migrated with verified identical weights).

For a new FlowPipe target, first generate its context cache:

```bash
python baselines/FlowPipe/cache_flowpipe_context.py \
  --data_names default_credit --data_dir /path/to/datasets \
  --output_dir outputs/flow_cache --llama_model_path /path/to/Qwen3-8B --gpu_id 0
```

Then pass `--context_cache_dir outputs/flow_cache/FlowPipe/context_embeddings`
to evaluation. Seen-source tasks already have caches in the checkpoint directory,
which evaluation selects by default. Regenerate embeddings after changing the
dataset, sampling seed/settings, encoder or quantization.

Results retain the original final standard-test metrics and add `training_mode`,
checkpoint/source provenance, seen-task flag, `training_seconds`, and
HAIPipe `scorer_adaptation_seconds`. Pretraining cost remains in `pretrain.json`
and is not charged again to every deployment.

## CPU verification

```bash
python -m unittest scripts.test_baseline_training \
  baselines.CtxPipe.test_context baselines.FlowPipe.test_flowpipe_alignment -v
```

These checks cover branch routing, checkpoint identity/rejection, actual shared
CtxPipe updates, gate round-trip, stable action IDs, frozen inference and exclusion
of frozen-test artifacts. Full dataset training, GPU feasibility and quality must
be measured on the experiment server.
