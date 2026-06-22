# ReAct Baseline

A faithful implementation of **ReAct** (Yao et al., ICLR 2023:
*"ReAct: Synergizing Reasoning and Acting in Language Models"*) for the
dppbench preprocessing-pipeline benchmark.

The agent runs a single linear Thought -> Action -> Observation trajectory.
At every turn the LLM submits a **complete preprocessing pipeline YAML**
(matching the schema of `dppbench/tasks/<task>/pre_process.yaml`); the
system resets the sandbox, executes the full pipeline, trains the
downstream model on a small validation sample, and feeds back

  * `status` (success / parse_error / sandbox_error / eval_error / legality_error),
  * the parsed op sequence,
  * the schema after pipeline execution,
  * the **full validation-set metrics dict** the trainer reported
    (e.g. `auc`, `logloss`, `acc`, `mse`, ...),
  * the primary `downstream_fitness`,
  * the running best (`fitness`, `turn`, `val_metrics`),

inside an `<observation>` tag. The LLM may then revise the YAML in the
next turn, or emit `<action>Terminate</action>` to stop. The
**highest-fitness** YAML across all turns is kept as the final pipeline.

## Files

```
baselines/ReAct/
  __init__.py          # exposes the ReAct top-level class
  operator_catalog.py  # thin re-export of the SAGA 77-op catalog
  evaluator.py         # ReActEvaluator(CtxPipeEvaluator) + evaluate_for_agent
  prompts.py           # SYSTEM_REACT + render_user_initial / render_observation / render_retry_feedback
  agent.py             # ReActAgent main loop (one full pipeline YAML per turn)
  react.py             # Top-level orchestrator (build sandbox/evaluator/llm, run, persist)
  run_react.py         # CLI entry point
  README.md            # this file
```

## CLI examples

```bash
# Tabular smoke test (API credentials are loaded from project apikeys.json)
python -m baselines.ReAct.run_react \
    --data_name fraud_detection --llm_model deepseek-v4-flash \
    --max_turns 3 --small_n 1000 --downstream_eval_n 500 --no_eval_full

# Recommendation smoke test
python -m baselines.ReAct.run_react \
    --data_name movielens --llm_model deepseek-v4-flash \
    --max_turns 3 --small_n 2000 --downstream_eval_n 1000 --no_eval_full
```

## Key hyperparameters

| Flag | Default | Purpose |
|---|---|---|
| `--max_turns` | 6 | Maximum full-pipeline turns the LLM may submit. |
| `--max_retry_per_turn` | 2 | Same-turn retry budget on parse / format errors. |
| `--max_err_cnt` | 5 | Cumulative cross-turn error budget. |
| `--downstream_eval_n` | 3000 | Subsample size for the in-loop downstream training. |
| `--small_n` | 0 | Sandbox subsample (0 = full data). |
| `--no_eval_full` | off | Skip the post-loop full-data evaluation. |

## Outputs (under `outputs/ReAct/<data_name>/`)

  * `best_pipeline.yaml` — berka-style YAML; the highest-fitness pipeline
    across all turns (with a final structural repair as a safety net).
  * `agent_log.json` — full LLM transcript + meta information.
  * `trajectory.json` — per-turn `TurnRecord` dump (thought, raw YAML,
    parsed ops, status, error, fitness, val metrics, observation text).
  * `run_summary.json` — duration, best turn, best fitness, downstream
    metrics, etc.

## Mapping to the ReAct paper

| Paper concept | This implementation |
|---|---|
| Thought | `<thought>` tag |
| Action | `<pipeline>YAML</pipeline>` or `<action>Terminate</action>` |
| Action space | dppbench 77 operators + `Terminate` |
| Observation | sandbox + downstream-trainer feedback (status, schema, full val-set metrics, fitness, running best) |
| Trajectory | `transcript` + `trajectory.json` |
| Termination | LLM-issued `Terminate` or `max_turns` reached |
