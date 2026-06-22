# BAT (Target-Instance-Free Data Preparation Synthesizer)

Re-implementation of `ZJU-DAILY/BAT` for dppbench. BAT (Bayesian-explored
Action Tree) drives an LLM through Monte Carlo Tree Search over a
constrained "data-preparation action sandbox" (DPAS) and picks the best
synthesized pipeline. This implementation also wires BAT's reward to a
**downstream ML model** (LightGBM/DIN), so the same code can be compared
fairly to the other LLM-driven baselines (DeepPrep, DataMaster) on
dppbench's tabular and recommendation tasks.

## Algorithmic Overview

BAT is built from three parts:

1. **DPAS** (Data Preparation Action Sandbox). Search-space restriction
   based on five node types and their legal successor actions:

   | Node type | Legal next actions |
   |---|---|
   | `ROOT` | `SchemaMatch`, `IdentifyColumnFunctions`, `Transformation` |
   | `SCHEMA_MATCH` | `IdentifyColumnFunctions`, `Transformation` |
   | `IDENTIFY_COLUMN_FUNCTIONS` | `SchemaMatch`, `Transformation` |
   | `TRANSFORMATION` | `End` (only if `columns_match`) or `End` + `TransformationRevision` |
   | `REVISED_TRANSFORMATION` | `End` |

   Action classes already used on the path-to-root are filtered out to
   avoid loops.

2. **FPG** (Fundamental Pipeline Generator). Standard MCTS:
   `Select` (UCB1, `Q/N + c·sqrt(ln(N_parent)/N_child)`),
   `Expand` (each legal action calls the LLM and adds children),
   `Simulate` (random rollout to `END`),
   `Backpropagate` (single reward at `END` walks up the path).
   Early stop after two paths reach reward `≥ 1 − ε`.

3. **EPO** (Execution-aware Pipeline Optimizer). At an `END` node BAT
   actually executes the final pipeline and computes the reward:
   - Schema-only mode (BAT original):
     `column_similarity = |actual ∩ expected| / |actual ∪ expected|`
     where `expected` is built from `DataContext` (no target instances
     needed -- "target-instance-free").
   - **dppbench mode (this implementation, default)**: weighted fusion
     of column similarity, downstream-model fitness, and an optional
     LLM-judge term:
     `reward = α·column_similarity + β·downstream_fitness + γ·llm_judge`
     (default `α=0.4, β=0.5, γ=0.1`).

The downstream channel uses small-N (default 3 000) subsampled training
to control cost, identical to DeepPrep / DataMaster. Setting
`--no_use_downstream` recovers the original BAT semantics exactly.

## Comparison with existing baselines

| Dimension | BAT (this impl.) | DeepPrep | DataMaster | SAGA |
|---|---|---|---|---|
| Decision agent | LLM + MCTS (DPAS) | LLM tree-search agent | LLM + UCB MCTS (black) | Evolutionary + physical tuning |
| Output form | Operator chain | Operator chain | Operator chain | YAML pipeline |
| Operator pool | dppbench 77 ops | dppbench 77 ops | dppbench 77 ops | own catalog (77) |
| Search constraint | DPAS 5-type DAG | tag protocol + repair | UCB on black nodes | category-order repair |
| Exploration feedback | schema (+ optional dwn.) | schema/dtype/sample | schema | direct eval |
| Final/select reward | column_sim + downstream | downstream AUC (small-N) | downstream AUC (small-N) | downstream AUC |
| Uses downstream during search | **yes** (`β`) | only after `<solution>` | every UCB | every fitness |
| Tabular | yes | yes | yes | yes |
| Rec | yes | yes | yes | yes |

## File layout

```
baselines/BAT/
├── __init__.py
├── README.md                    # this file
├── operator_catalog.py          # 77-op whitelist + function-family map
├── prompts.py                   # SchemaMatch / IdentifyFunctions /
│                                # Transformation / Revision / Reward
├── action.py                    # 5 action classes + legality table
├── node.py                      # MCTSNode + MCTSNodeType
├── reward.py                    # column_sim + downstream + llm-judge
├── mcts.py                      # MCTSSolver
├── evaluator.py                 # BATEvaluator (LightGBM / DIN AUC)
├── sandbox.py                   # re-export of DeepPrep Sandbox
├── bat.py                       # top-level orchestrator
└── run_bat.py                   # CLI
```

## Quick start

Tabular smoke test (downstream feedback ON, the dppbench default):

```bash
# API credentials are loaded from project apikeys.json.
python -m baselines.BAT.run_bat \
    --data_name fraud_detection --llm_model gpt-4o-mini \
    --max_rollout_steps 4 --max_depth 4 --small_n 3000
```

Tabular smoke test (downstream OFF, recovers BAT-original behaviour):

```bash
python -m baselines.BAT.run_bat \
    --data_name fraud_detection --llm_model gpt-4o-mini \
    --max_rollout_steps 4 --max_depth 4 --small_n 3000 \
    --no_use_downstream
```

Recommendation smoke test (mandatory ops are added by `repair`):

```bash
python -m baselines.BAT.run_bat \
    --data_name movielens --llm_model gpt-4o-mini \
    --max_rollout_steps 4 --max_depth 4 --small_n 3000
```

## Outputs

Every run writes the following under
`outputs/BAT/<data_name>/`:

| File | Content |
|---|---|
| `best_pipeline.yaml` | repaired final pipeline ready for `dppbench.dataset.run_pre_process` |
| `agent_log.json` | rollout / best-node / reward breakdown summary |
| `tree.json` | full search tree (for analysis & visualisation) |
| `best_paths.json` | paths that reached reward `≥ 1 − ε` |

## Trade-off note

BAT was originally proposed in a *target-instance-free* setting where
only the source data + target schema is available, so it never trains a
downstream model. dppbench requires a downstream LightGBM (tabular) or
DIN (rec) at the end of every pipeline. To bridge the two, this
implementation:

* Keeps BAT's column-similarity term (`α`) so the original signal is
  preserved.
* Adds a downstream-fitness term (`β`) to satisfy dppbench's request to
  feed back the downstream metric.
* Provides `--no_use_downstream` for users who want to reproduce the
  paper-faithful behaviour without the downstream channel.
