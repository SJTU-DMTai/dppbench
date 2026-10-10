# HAIPipe baseline for DPPBench

This implementation trains one HAIPipe instance independently for every
`task × downstream model` run. It preserves AIPipe reinforcement learning, HAI
enumeration, and active-learning selection. DPPBench's shared operator space and
YAML/DAG executor replace the upstream notebook runtime.

## Method mapping

| HAIPipe concept | DPPBench implementation |
|---|---|
| HI-pipeline | The task's human-authored `prepare.yaml` |
| AI-pipeline | Logical + component DQNs trained on the current task/model |
| HAI-pipeline | Ordered-subset insertion enumeration over the HI DAG |
| Candidate selection | HAIPipe alternative evaluator and iterative AL |
| Legality | `finalize_pipeline(...)` and `check_pipeline(...)` |

A normal benchmark run starts both trainable parts from random initialization.
The AIPipe policy is trained against the current task/model validation objective,
then the alternative evaluator is fitted online from the actively selected HAI
candidates. The resulting `aipipe.pt` and `alternative_evaluator.pt` files are
saved under that run's output directory for audit and reproduction. Standard-test
data remains unavailable during both training phases.

The published upstream weights are not used by this default path because
DPPBench's operator vocabulary and task/model protocol differ. They still require
an explicit vocabulary/component mapping before they can be used in a separate
transfer-learning experiment.

## Common API usage

The implementation relies on shared APIs for all executable details:

- `get_data_ctx(...)`
- `get_available_ops(...)`
- `make_step(...)`
- `finalize_pipeline(...)`
- `check_pipeline(...)`
- `SearchPipelineEvaluator(...)`

It does not maintain a private operator catalog or private parameter factory.

## Files

- `haipipe.py`: top-level runner and output contract
- `hi_pipeline.py`: HI scaffold construction
- `operator_mapping.py`: HAIPipe components to DPPBench categories
- `env.py`: component-level AI-pipeline environment
- `trainer.py`: DQN replay updates at interaction-step intervals
- `combiner.py`: HI + AI enumeration/sampling and best candidate selection

## Optional cross-task pretraining

`baselines/HAIPipe/pretrain_haipipe.py` remains available for ablation or transfer-learning
experiments. It is not required by the default baseline runner. A cross-task run
must receive an explicit training-task split; do not include any target/test task
in `--data_names`:

```bash
python baselines/HAIPipe/pretrain_haipipe.py \
  --data_names <training tasks> \
  --held_out_data_names <target/test tasks> \
  --model <model> \
  --data_dir /mnt/sdb/dengjiale/dppbench/data \
  --output_dir /mnt/sdb/dengjiale/dppbench/output \
  --gpu_id <idle GPU>
```

The command writes `aipipe.pt` and `alternative_evaluator.pt` below
`<output_dir>/HAIPipe/pretrain/<model>/`. The default per-task runner does not
load them. Pretraining is a long-running GPU job and should normally be launched
in tmux after checking the selected GPU.

## Model alignment and necessary adaptations

References: [paper, Sections 4–6](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/haipipe-acm.pdf)
and upstream [commit 34d2cc8](https://github.com/ruc-datalab/Haipipe/tree/34d2cc8d24c47f0a8b3f6f19eeab2d20e4e71061).
The paper describes the architecture at a high level. Exact dimensions and
parameter names follow `aipipe/core/agent/model.py`; the alternative evaluator
follows the **released `model/10` checkpoint**, which has additional BatchNorm
layers absent from upstream's current `model.py`.

| Network | Preserved upstream structure | Operator-library adaptation |
|---|---|---|
| Logical DQN | 1901 inputs; 11 linear layers; LeakyReLU and final Tanh | Output has C! logical orders for C nonempty shared categories |
| Component DQN | 30-d operator embeddings; 2-layer 18-d LSTM; 10-d model and 5-d logical embeddings; upstream MLP widths/names | C sequence slots, shared operator/action vocabularies, one configured downstream model |
| Alternative evaluator | 1900→950→950→300→300 data branch; 30-d embeddings; 5-layer 100-d LSTM; 30 slots; 3300→1650→200→200→10→1 head with BatchNorm | Operator embedding row count only |

`upstream_checkpoint_shapes.json` records parameter/buffer shapes extracted from
`56000_logical_pipeline.pkl`, `56000_fpreprocessing_model.pkl`, and `model/10`.
The regression test instantiates our networks with the original vocabulary sizes
and checks every key and shape against these independent snapshots.

State semantics:

- Recompute the 100×19 statistics after each successful operator using the shared
  partial-DAG executor. Exclude the target column, standard-test rows, and split
  marker. Match upstream's nonmissing mode frequency and boolean treatment.
- The DQN sequence records one slot per visited component, including skipped
  components. Operator IDs are 1..N, blank is 0, and unvisited padding is N+1.
  The logical DQN uses a one-based model ID; the component embedding is zero-based.
- The evaluator uses padding 0 and a distinct unknown-operation token N+1.
- Greedy DQN action inference disables dropout; learning still enables it.

DQN training retains Adam lr=1e-5, replay capacity 2000, component batch size 100,
logical batch size 20, and epsilon 1.0→0.4 with decay 500. Bootstrap values come
from the **next component's** network, with zero bootstrap at termination.
The effective upstream Bellman coefficient is 1 (its `learning()` ignores the
`gamma=0` declaration), so this adapter explicitly uses `gamma=1`.
Updates run every 100 interaction steps after replay warmup, even when an episode
crosses an update boundary. Final evaluation adds mandatory operators through
`finalize_pipeline(sort_steps=False)` without sorting away the chosen order.

Alternative-evaluator offline training matches `haipipe/core/al/pretrain.py`:
Adam lr=0.01, batch size 16, 10 epochs, L1 loss and learning-rate multiplier 0.9
at the start of each epoch. A final singleton batch is merged into its predecessor
to support BatchNorm without discarding an example. Target-task fine-tuning keeps
lr=0.001 and five epochs. The pretraining CLI reads training defaults from
`config.yaml` and accepts explicit overrides. The default DPPBench DQN budget is 56000 frames, matching the paper's reported
training count.

The mandatory DPPBench adaptations remain explicit: four shared categories replace
the original six-stage layout; parameterized operators such as ScaleFeature remain
single shared actions; the model vocabulary contains the selected downstream model;
and shared YAML/DAG legality, validation fitness, and standard-test replay remain
in force. We do not add private sklearn operators, expand search budgets, or alter
shared downstream models/splits. For the default runner, policy learning and
active candidate evaluation are both part of the task/model baseline training cost.

Both checkpoint contracts are now version 2. Version-1 DPPBench checkpoints are
rejected because state/token semantics and the evaluator architecture changed.
They require retraining or an explicit migration; changing metadata alone is not
valid. Upstream checkpoints likewise need mapping and training-task provenance,
not just matching shapes.
