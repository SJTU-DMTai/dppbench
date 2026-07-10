# DeepPrep Baseline

LLM-powered, tree-based agentic data preparation adapted to DPPBench. The
implementation keeps DeepPrep's inference-time structure — execution-grounded
planning, tree expansion, materialized state snapshots, parent-node selection,
and backtracking — while using DPPBench's shared operator space, DAG format,
parameter validation, legality checks, and downstream objective.

## Method mapping

| Aspect | DPPBench DeepPrep |
|---|---|
| Decision policy | LLM agent (API by default; local HF transformers optional) |
| One-step output | `<plan>` + `<operator>` / `<solution>` / `<backtrack/>` |
| Parent selection | Optional `<operator parent="OpA --> OpB">...</operator>` prefix selection |
| Tree state | One node per materialized sandbox state; one edge per operator instance |
| Exploration feedback | Structural sandbox feedback only: schema, dtype, sample rows, errors |
| Solution feedback | Optional small-data DPPBench downstream fitness for best-of-N attempts |
| Operator pool | Shared `baselines.common` operators for the current task type |
| Final scoring | Shared DPPBench evaluator: LightGBM/tabular or DIN/rec AUC |
| Training | Inference-only by default; SFT/GRPO/hybrid-reward training remains a stub |

Compared with the original DeepPrep paper, the benchmark adapter intentionally
changes only the operator space and optimization target: operators come from
DPPBench common APIs, and reward/selection uses downstream ML fitness rather
than target-table exact match. The progressive training framework from the
paper (cold-start SFT, GRPO, hybrid reward, data synthesis) is not implemented
in the default baseline path.

## Common API usage

Executable details are delegated to common APIs:

- `get_data_ctx(...)`
- `get_available_ops(...)`
- `make_step(...)`
- `validate_step_params(...)`
- `assign_dag_structure(...)`
- `finalize_pipeline(...)`
- `check_pipeline(...)`
- `SearchPipelineEvaluator(...)`
- `subsample_data_inplace(...)`

The baseline does not maintain a private operator parameter generator or a
private YAML/DAG implementation.

## File map

```text
baselines/DeepPrep/
├── __init__.py
├── operator_catalog.py       # renders common operator metadata for prompts
├── llm_client.py             # API + local HF backends
├── prompts.py                # system/user/observation templates
├── sandbox.py                # common-DAG incremental execution + snapshots
├── tree_node.py              # SearchNode / SearchTree
├── tree_agent.py             # tree reasoning loop and chain parser
├── evaluator.py              # SearchPipelineEvaluator alias
├── rl_trainer.py             # optional training scaffold
├── deepprep.py               # top-level orchestrator
└── README.md
```

## Quick start

```bash
python scripts/evaluate_baseline.py --baseline DeepPrep --data_names amazon_beauty --model DIN
```

## Configuration (`config.yaml`)

| key | default | description |
|---|---:|---|
| `llm_backend` | `api` | `api` or `local` |
| `llm_model` | `gpt-4o-mini` | model name or local model path |
| `temperature` | `0.7` | LLM sampling temperature |
| `max_tokens` | `2048` | per-call generation budget |
| `max_explore_turn` | `5` | maximum exploration turns before forcing a solution |
| `max_chain_len` | `6` | max operators per action and final pipeline budget |
| `max_depth` | `8` | max search-tree depth |
| `max_err_cnt` | `5` | abort after this many agent errors |
| `small_n` | `0` | sandbox subsample size; `0` disables |
| `eval_full` | `true` | re-evaluate the best pipeline on full data in baseline run |
| `downstream_feedback` | `true` | enable in-loop downstream feedback after successful solutions |
| `downstream_eval_n` | `3000` | small-N budget for in-loop downstream evaluator |
| `max_solution_attempts` | `3` | max solution attempts with feedback |
| `seed` | `42` | RNG seed |
| `fast_train` | `true` | use fast train overrides in search-time evaluators |

## Outputs

`outputs/DeepPrep/<data_name>/`

- `best_pipeline.yaml` — final common-finalized DAG pipeline.
- `agent_log.json` — assistant turns, solution attempts, validation results.
- `tree.json` — materialized search tree nodes, parents, ops, and errors.
