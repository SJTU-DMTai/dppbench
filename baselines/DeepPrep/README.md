# DeepPrep Baseline

LLM-powered, tree-based agentic data preparation. The agent constructs a
pipeline from scratch by repeatedly proposing operator chains. During
exploration (`<operator>`) it observes only sandbox structural feedback
(schema, sample rows, errors). When `downstream_feedback` is enabled
(default), every successful `<solution>` attempt also receives a
small-data downstream-model metric (LightGBM/DIN AUC) so the LLM can
iteratively refine the pipeline. The system keeps the attempt with the
highest small-data fitness and re-evaluates it on the full data at the
end.

## Method overview

| Aspect          | DeepPrep                                                                 |
|-----------------|--------------------------------------------------------------------------|
| Decision policy | LLM agent (API by default; local HF transformers optional)               |
| One-step output | `<plan>` / `<operator>` chain / `<solution>` / `<backtrack/>`            |
| Feedback        | `<operator>`: sandbox schema/dtype/sample rows/error trace.               |
|                 | `<solution>`: above + small-data downstream AUC (best-of-N attempts).     |
| Search shape    | Tree over sandbox snapshots; non-local backtracking                      |
| Operator pool   | All 27 operators in `dppbench/ operators/` (tabular + recommendation)    |
| Final scoring   | LightGBM AUC (tabular) / DIN AUC (rec) — both in-loop (small-N) AND end |
| Training        | Inference-only by default. RL stub (`rl_trainer.py`) preserved           |

## File map

```
baselines/DeepPrep/
├── __init__.py               # exposes DeepPrep
├── operator_catalog.py       # re-exports SAGA's 27-operator catalog
├── llm_client.py             # API + local HF backends (lazy local import)
├── prompts.py                # system / user / observation / backtrack templates
├── sandbox.py                # incremental operator execution + snapshots
├── tree_node.py              # SearchNode / SearchTree
├── tree_agent.py             # tree-based reasoning loop (parse + run)
├── evaluator.py              # final-pipeline downstream-model scorer
├── rl_trainer.py             # RL training stub (NotImplementedError by default)
├── deepprep.py               # top-level orchestrator
└── run_deepprep.py           # CLI entry point
```

## Quick start

### 1. Tabular smoke test (LLM API)

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib \
python -m baselines.DeepPrep.run_deepprep \
    --data_name fraud_detection --llm_model gpt-4o-mini \
    --max_explore_turn 3 --max_chain_len 4 --small_n 3000
```

### 2. Recommendation smoke test (LLM API)

```bash
python -m baselines.DeepPrep.run_deepprep \
    --data_name movielens --llm_model gpt-4o-mini \
    --max_explore_turn 3 --max_chain_len 4 --small_n 3000
```

### 3. Local backend (optional)

```bash
python -m baselines.DeepPrep.run_deepprep \
    --data_name fraud_detection \
    --llm_backend local --llm_model Qwen/Qwen3-0.6B
```

`backend="local"` lazily imports `transformers` only when `.chat()` is
called, so installing the OpenAI SDK alone is enough for the API path.

## CLI flags

| flag                  | default       | description                                                |
|-----------------------|---------------|------------------------------------------------------------|
| `--data_name`         | `movielens`   | one of the six benchmark datasets                          |
| `--llm_backend`       | `api`         | `api` or `local`                                           |
| `--llm_model`         | `gpt-4o-mini` | model name (api) or HF repo / local path (local)           |
| `--api_key`           | disabled      | credentials are loaded from project `apikeys.json`         |
| `--base_url`          | disabled      | endpoint is loaded from project `apikeys.json`             |
| `--temperature`       | `0.7`         | sampling temperature                                       |
| `--max_tokens`        | `2048`        | per-call generation budget                                 |
| `--max_explore_turn`  | `5`           | maximum exploration turns before forcing `<solution>`      |
| `--max_chain_len`     | `6`           | max operators per `<operator>` chain                       |
| `--max_depth`         | `8`           | max search-tree depth                                      |
| `--max_err_cnt`       | `5`           | abort the agent after this many errors                     |
| `--small_n`           | `0`           | sandbox subsample size (`0` = full data)                   |
| `--no_eval_full`      | off           | skip the downstream-model evaluation at the end            |
| `--no_downstream_feedback` | off      | disable the in-loop downstream training feedback after each `<solution>` (legacy: accept first successful solution) |
| `--downstream_eval_n` | `3000`        | subsample size used by the agent-loop downstream evaluator (`0` = full data) |
| `--max_solution_attempts` | `3`       | max number of `<solution>` attempts when downstream feedback is enabled |
| `--seed`              | `42`          | RNG seed for default-param synthesis                       |
| `--output_dir`        | auto          | where to dump pipeline / agent log / search tree           |
| `--quiet`             | off           | silence progress prints                                    |

## Outputs

`outputs/DeepPrep/<data_name>/`

* `best_pipeline.yaml` — final repaired pipeline.
* `agent_log.json` — every assistant turn + per-`<solution>` attempt
  metadata (ops, downstream fitness, metrics, error). Also records
  `downstream_feedback`, `downstream_eval_n`, and `max_solution_attempts`.
* `tree.json` — the search tree (nodes, parents, ops, errors).

## Switching to local inference / RL

* Default verification path is **API only** — no training happens.
* To run a local model, pass `--llm_backend local --llm_model <path>`.
* To enable RL, subclass `RLTrainer` (see `rl_trainer.py`):
  * `collect_trajectory(task)` returns trajectories whose reward, by
    default (`use_downstream_reward=True`), is the best small-data
    downstream metric across the agent's `<solution>` attempts. Pass
    `use_downstream_reward=False` to recover the legacy binary reward.
  * Override `train(trajectories)` to plug in your gradient update;
    share weights with `LLMClient.attach_local_model(model, tokenizer)`.

## Design notes

* During `<operator>` exploration the agent only sees sandbox structural
  feedback (schema, dtype, sample rows, error trace). Downstream-model
  training is NEVER triggered during exploration.
* When `downstream_feedback=True` (default), every successful `<solution>`
  attempt triggers a small-data (`--downstream_eval_n`) downstream
  training; the resulting AUC is fed back to the LLM via an
  `<observation>SOLUTION ATTEMPT k/N evaluated...</observation>` message.
  The LLM may submit up to `--max_solution_attempts` `<solution>` chains;
  the system keeps the highest-scoring one as the final pipeline and
  re-evaluates it on the full data when `--no_eval_full` is not set.
* Operator parameters are auto-filled with `build_default_params(...)`,
  so the LLM only needs to write `OpName(arg=value)` (or just `OpName`).
* After the agent emits a `<solution>`, the pipeline is passed through
  `pipeline_constraints.repair(...)`. This guarantees mandatory operators
  are present (rec: `JoinTable`, `CreateSequence`,
  `DataSplit`; tabular: `LabelEncode`+`HandleMV` tail) so the final
  evaluator always receives an executable pipeline.
* Snapshots use `pickle` over the dataset's `BaseData` instance. Use
  `--small_n` for very large datasets to keep snapshots manageable.
