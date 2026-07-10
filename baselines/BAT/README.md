# BAT Baseline

DPPBench adapter for **BAT: Target-Instance-Free Data Preparation Synthesis via LLM-driven Tree Search**. The implementation preserves BAT's inference-time structure — DPAS action sandbox, LLM-driven MCTS, and execution-aware pipeline optimization — while using DPPBench's shared operator space, common DAG/legality contract, and downstream ML objective.

## Method mapping

| BAT paper concept | DPPBench implementation |
|---|---|
| DPAS action sandbox | `SchemaMatch`, `IdentifyColumnFunctions`, `Transformation`, `TransformationRevision`, `End` node/action types |
| FPG | UCB-style MCTS in `mcts.py`; each legal BAT action expands the tree |
| EPO | Sandbox execution of transformation candidates plus column-similarity feedback |
| Target-instance-free reward | Column similarity against `DataContext`-derived expected schema |
| DPPBench objective | Optional downstream AUC term from `SearchPipelineEvaluator` |
| Operator/action language | Shared DPPBench operators rendered from `baselines.common.OpSpec` |
| Final executable format | Common prev-only DAG YAML via `finalize_pipeline(...)` |

The original BAT paper uses source tables and target schema without target table instances. In DPPBench, the target schema is approximated from `DataContext`, the operator library is the shared DPPBench operator space, and downstream fitness can be fused into the reward for benchmark comparability. Set `use_downstream: false` to recover the schema-only BAT-style reward.

## Algorithmic overview

1. **DPAS** restricts search to five action/node types:

   | Node type | Legal next actions |
   |---|---|
   | `ROOT` | `SchemaMatch`, `IdentifyColumnFunctions`, `Transformation` |
   | `SCHEMA_MATCH` | `IdentifyColumnFunctions`, `Transformation` |
   | `IDENTIFY_COLUMN_FUNCTIONS` | `SchemaMatch`, `Transformation` |
   | `TRANSFORMATION` | `End`, and `TransformationRevision` when EPO feedback is not sufficient |
   | `REVISED_TRANSFORMATION` | `End` |

2. **FPG** performs MCTS with UCB selection, action expansion, random rollout to `END`, and reward backpropagation.
3. **EPO** executes transformation candidates in the shared sandbox, records `columns_match`, `column_similarity`, and `exec_error`, and feeds these signals to revision prompts.
4. **Finalization** sends every terminal/final pipeline through common `finalize_pipeline(...)`, which repairs mandatory operators, truncates optional steps to the budget, assigns DAG structure, and runs `check_pipeline(...)`.

## Common API usage

BAT delegates executable details to common APIs instead of maintaining private implementations:

- `get_data_ctx(...)`
- `get_available_ops(...)`
- `make_step(...)` / `validate_step_params(...)` through DeepPrep `chain_to_steps(...)`
- `finalize_pipeline(...)`
- `check_pipeline(...)`
- `SearchPipelineEvaluator(...)`
- DeepPrep `Sandbox` for common DAG execution and small-N sampling

## Files

```text
baselines/BAT/
├── __init__.py
├── action.py             # DPAS actions and legality table
├── bat.py                # top-level runner
├── config.yaml           # default hyperparameters
├── evaluator.py          # SearchPipelineEvaluator alias
├── mcts.py               # UCB MCTS driver
├── node.py               # MCTS node data structure
├── operator_catalog.py   # common OpSpec prompt rendering
├── prompts.py            # BAT prompt templates
├── reward.py             # column/downstream/LLM reward fusion
└── sandbox.py            # DeepPrep Sandbox re-export
```

## Quick start

```bash
python scripts/evaluate_baseline.py --baseline BAT --data_names amazon_beauty --model DIN
```

## Key configuration

| key | default | description |
|---|---:|---|
| `max_rollout_steps` | `10` | MCTS rollout budget |
| `max_depth` | `5` | MCTS tree depth budget |
| `max_chain_len` | `6` | final pipeline length budget passed to common finalization |
| `n_action_samples` | `1` | LLM samples per expandable BAT action |
| `exploration_constant` | `1.0` | UCB exploration coefficient |
| `reward_alpha` | `0.4` | column-similarity reward weight |
| `reward_beta` | `0.5` | downstream-fitness reward weight |
| `reward_gamma` | `0.1` | optional LLM-judge reward weight |
| `use_downstream` | `true` | include downstream reward during search |
| `use_llm_judge` | `false` | include LLM reward judge |
| `small_n` | `0` | sandbox subsample size |
| `downstream_eval_n` | `3000` | small-N downstream reward budget |
| `eval_full` | `true` | run final full-data evaluation inside baseline run |

## Outputs

`outputs/BAT/<data_name>/` contains:

- `best_pipeline.yaml` — common-finalized DAG pipeline.
- `agent_log.json` — rollout count, best node, reward breakdown, validation report.
- `tree.json` — full MCTS tree.
- `best_paths.json` — terminal paths reaching near-perfect reward.
