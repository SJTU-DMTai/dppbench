"""Prompt templates for the ReAct agent (full-pipeline-YAML protocol).

Every turn the LLM emits a *complete* pipeline YAML inside ``<pipeline>``,
optionally preceded by ``<thought>``. To finish, it emits
``<action>Terminate</action>`` instead of a new ``<pipeline>``.

The ``<observation>`` we return after each pipeline turn includes:
  * status (success / parse_error / sandbox_error / eval_error / legality_error)
  * the parsed op sequence (for echo feedback)
  * sandbox schema text after executing the pipeline
  * the *full* metrics dict the downstream trainer reported on the
    validation / held-out eval split (auc, logloss, acc, mse, ...)
  * downstream_fitness (the primary metric used for ranking)
  * the running best across turns (fitness, turn id, full val metrics)
  * how many turns are left
"""
from __future__ import annotations

from typing import Optional


SYSTEM_REACT = """You are ReAct, a data-preparation agent that builds a
preprocessing pipeline iteratively under a Thought -> Action -> Observation
loop.

# Action protocol
Each turn you MUST output exactly ONE action. Two action shapes are allowed:

  (A) submit a new pipeline:
      <thought>your reasoning about what to keep/change vs the previous attempt</thought>
      <pipeline>
      pipeline:
        - op: OpName1
          target: ...
          params:
            ...
        - op: OpName2
          target: ...
          params:
            ...
      </pipeline>

  (B) terminate the loop and accept the best version so far:
      <thought>The pipeline is final.</thought>
      <action>Terminate</action>

The content INSIDE <pipeline>...</pipeline> MUST be valid YAML that follows
the EXACT schema of `dppbench/tasks/<task>/pre_process.yaml`:

    pipeline:
      - op: <OperatorName>
        target: <interaction|train|test|both>
        params:
          <key>: <value>

The pipeline you write each turn is the FULL pipeline (NOT a delta on top
of the previous one). The system will:
  1. Parse the YAML into a Pipeline object.
  2. Reset the sandbox to the original data.
  3. Execute your full pipeline on the sandbox.
  4. Train the downstream model on a small validation sample and report
     ALL metrics it produces (e.g. auc / logloss / acc / mse).
  5. Reply with an <observation> containing status, schema, the val-set
     metrics dict, the primary downstream_fitness, your best fitness so
     far (with the corresponding turn id and its full val metrics), and
     how many turns are left.

You may then revise the YAML for the next turn. To stop, emit
<action>Terminate</action> instead of <pipeline>. The system keeps the
highest-fitness attempt across all turns as the final pipeline.

# Constraints
- You may ONLY use operators from the whitelist injected in the user message.
- You may NOT invent operator names or new arguments.
- You may NOT write Python code; the only way to act on the data is via
  the operator chain.
- Parameters that depend on the dataset (column lists, target columns, etc.)
  may be omitted; the system will fill them with safe defaults.
- The accumulated pipeline MUST contain operators in non-decreasing
  category-rank order (the "Canonical Category Order" in the user message).
  Out-of-order pipelines are rejected with a legality_error.
- Reflect on the validation-set metrics from the previous observation when
  deciding what to change in the next pipeline.
"""


def render_user_initial(
    *,
    task_type: str,
    data_name: str,
    summary_text: str,
    op_descriptions: str,
    obs_text: str,
    yaml_example: str,
    ordering_hint: str,
    max_turns: int,
) -> str:
    target_text = (
        "downstream LightGBM (binary AUC; metrics on the validation split)"
        if task_type == "tabular"
        else "downstream DIN recommender (AUC; metrics on the held-out eval split)"
    )
    return f"""# Task
Construct a data preparation pipeline for the **{task_type}** dataset
`{data_name}`. The pipeline you submit each turn will be executed on the
data and the resulting tables will be consumed by a {target_text}.

# Dataset Summary
{summary_text}

## Sandbox Observation (state of the original data BEFORE any operator is applied)
{obs_text}

# Operator Whitelist
Use ONLY these operators (grouped by category). Each entry shows whether
it is mandatory, the task type it applies to, and its parameter names.
{op_descriptions}

# Canonical Category Order (HARD constraint)
Your full pipeline MUST contain operators in non-decreasing category-rank
order. The system rejects out-of-order pipelines with a legality_error.
Order is:
{ordering_hint}

# YAML Schema (FULL pipeline each turn)
Your <pipeline> block MUST be valid YAML matching the schema of
`dppbench/tasks/<task>/pre_process.yaml`. For reference, here is what a
hand-written reference pipeline looks like for a similar task:
```yaml
{yaml_example.strip()}
```

# Loop Budget
- max_turns: {max_turns}
- Each turn: emit a <thought> + a complete <pipeline> YAML, OR a <thought>
  + <action>Terminate</action> to accept the best attempt so far.
- The system feeds back the validation-set metrics dict every turn so you
  can reason about which parts of the previous pipeline to keep.

# This is turn 1
Submit the FIRST pipeline now. Reflect briefly inside <thought>...</thought>,
then emit one <pipeline>...</pipeline> block whose content is the YAML.
""".strip()


def _format_metrics(metrics: dict) -> str:
    if not metrics:
        return "(no metrics)"
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def render_observation(
    *,
    status: str,
    parsed_ops: list[str],
    schema_text: Optional[str] = None,
    error: Optional[str] = None,
    fitness: Optional[float] = None,
    val_metrics: Optional[dict] = None,
    best_fitness: Optional[float] = None,
    best_turn: Optional[int] = None,
    best_metrics: Optional[dict] = None,
    turn: int,
    max_turns: int,
) -> str:
    turns_left = max(0, max_turns - turn)
    fit_str = f"{fitness:.4f}" if isinstance(fitness, float) else "n/a"
    best_str = f"{best_fitness:.4f}" if isinstance(best_fitness, float) else "n/a"
    best_turn_str = str(best_turn) if best_turn is not None else "n/a"
    val_str = _format_metrics(val_metrics or {})
    best_val_str = _format_metrics(best_metrics or {})
    ops_str = (" -> ".join(parsed_ops)) if parsed_ops else "(none)"

    body_lines = [
        "<observation>",
        f"status: {status}",
        f"submitted_ops: {ops_str}",
    ]
    if error:
        body_lines.append(f"error: {error}")
    if schema_text:
        body_lines.append("schema_after_pipeline:")
        body_lines.append(schema_text)
    if val_metrics is not None:
        body_lines.append(f"val_metrics: {{{val_str}}}")
    body_lines.append(f"downstream_fitness: {fit_str}")
    body_lines.append(
        f"best_fitness_so_far: {best_str} "
        f"(turn={best_turn_str}, val_metrics={{{best_val_str}}})"
    )
    body_lines.append(f"turns_left: {turns_left}")
    if turns_left == 0:
        body_lines.append(
            "NOTE: this is the LAST observation. Reply with "
            "<thought>...</thought><action>Terminate</action> to stop."
        )
    body_lines.append("</observation>")
    return "\n".join(body_lines)


def render_retry_feedback(error: str, retries_left: int) -> str:
    return (
        "<observation>FORMAT ERROR.\n"
        f"reason: {error}\n"
        f"same-turn retries left: {retries_left}\n"
        "Reply again with the required tags. Either submit a new "
        "<pipeline>...</pipeline> YAML, or emit "
        "<action>Terminate</action> to stop.\n"
        "</observation>"
    )


__all__ = [
    "SYSTEM_REACT",
    "render_user_initial",
    "render_observation",
    "render_retry_feedback",
]
