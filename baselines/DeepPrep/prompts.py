"""Prompt templates for the DeepPrep tree-based agent.

These templates implement the tag-based protocol used in the original
DeepPrep paper: every assistant response should contain exactly ONE action
tag from {<operator>, <solution>, <backtrack/>}. Optional <plan>/<think>
tags may precede the action tag.

NOTE: During ``<operator>`` exploration the agent only sees sandbox
structural feedback (schema, sample rows, error traces). When downstream
feedback is enabled, every successful ``<solution>`` triggers a small-data
downstream training and the resulting metric is fed back to the LLM so it
can refine the pipeline before the budget is exhausted.
"""
from __future__ import annotations

from typing import Optional


SYSTEM_TREE_SEARCH = """You are DeepPrep, an LLM-powered data-preparation agent.
You construct an executable preprocessing pipeline by repeatedly proposing
short *operator chains* and observing the resulting intermediate table.

# Action protocol
Every reply MUST start with optional reasoning tags and end with EXACTLY ONE
action tag.

Reasoning tags (optional, may appear in any order before the action tag):
  <think>brief reasoning</think>
  <plan>high-level plan for the remaining pipeline</plan>

Action tags (MUST contain exactly one of):
  <operator>Op1(arg=value) --> Op2(arg=value) --> ...</operator>
      Append this chain to the current pipeline (executes in the sandbox).
      The chain length is bounded by max_chain_len. Use this to *explore*.
  <solution>Op1(...) --> Op2(...) --> ... --> Terminate</solution>
      Finalize. The chain you write is the FULL pipeline (replacing whatever
      was accumulated). It MUST end with `Terminate` and be self-contained.
  <backtrack/>
      Discard the most recent successful expansion and return to the parent
      search node. Use this when the current branch is clearly stuck.

# Constraints
- You may ONLY use operators from the whitelist injected in the user message.
- You may NOT invent operator names or new arguments.
- Parameters that depend on the dataset (column lists, target columns, etc.)
  may be omitted; the system will fill them with safe defaults.
- The ``<operator>`` feedback channel is purely structural (schema, sample
  rows, error traces). Downstream-model training is NEVER triggered during
  exploration.
- When you submit ``<solution>`` AND it executes successfully, the system
  trains the downstream model on a small subsample and replies with a
  ``downstream_fitness`` (e.g. AUC). You may then submit a refined
  ``<solution>`` to try to improve the metric, up to ``max_solution_attempts``
  times. The system keeps the attempt with the best fitness as the final
  pipeline (then re-evaluates it on the full data).
- If you see a repeated error, BACKTRACK or rewrite the chain — do not retry
  the same chain.
"""


def render_user_initial(
    task_type: str,
    data_name: str,
    summary_text: str,
    op_descriptions: str,
    obs_text: str,
    max_chain_len: int,
    max_explore_turn: int,
) -> str:
    target_text = (
        "downstream LightGBM (binary AUC)" if task_type == "tabular"
        else "downstream DIN recommender (AUC)"
    )
    return f"""# Task
Construct a data preparation pipeline for the **{task_type}** dataset
`{data_name}`. The dataset will eventually be consumed by a {target_text},
but the training itself is performed AFTER you submit your <solution>; it
is NOT part of your feedback loop.

# Dataset Summary
{summary_text}

# Initial Sandbox Observation
{obs_text}

# Operator Whitelist
Use ONLY these operators (grouped by category):
{op_descriptions}

# Limits
- Max operator-chain length per turn: {max_chain_len}
- Max exploration turns (excluding the final solution): {max_explore_turn}

Begin by reasoning about the dataset, then issue an `<operator>` chain to
explore, or directly emit a `<solution>` if the answer is obvious.
""".strip()


def render_observation(
    obs_text: str,
    error: Optional[str],
    turn_left: int,
    max_explore_turn: int,
) -> str:
    if error:
        body = (
            "<observation>EXECUTION FAILED.\n"
            f"error: {error}\n"
            "The current pipeline state was NOT updated. Either rewrite the "
            "chain (different ops or args), or emit <backtrack/>."
        )
    else:
        body = "<observation>EXECUTION OK.\n" + obs_text
    if turn_left <= 1:
        reminder = (
            f"This is the LAST exploration turn out of {max_explore_turn}. "
            "You MUST emit <solution>...</solution> next."
        )
    else:
        reminder = (
            f"You have {turn_left} more exploration turns. "
            "Use <operator> to keep exploring, <backtrack/> to revert, or "
            "<solution> to finalize."
        )
    body += "\n<reminder>" + reminder + "</reminder>"
    body += "</observation>"
    return body


def render_backtrack_hint(parent_obs: str, removed_ops: list[str]) -> str:
    removed = ", ".join(removed_ops) or "(nothing)"
    return (
        "<observation>BACKTRACKED. "
        f"Discarded ops: [{removed}]. Restored parent state.\n"
        + parent_obs
        + "</observation>"
    )


def render_solution_feedback(
    pipeline_ops: list[str],
    fitness: Optional[float],
    metrics: dict,
    error: Optional[str],
    attempt_idx: int,
    max_attempts: int,
    best_fitness: Optional[float],
) -> str:
    """Render the downstream-training feedback after a ``<solution>`` attempt."""
    attempts_left = max_attempts - attempt_idx
    fit_str = f"{fitness:.4f}" if isinstance(fitness, float) else "n/a"
    best_str = f"{best_fitness:.4f}" if isinstance(best_fitness, float) else "n/a"
    lines = [f"<observation>SOLUTION ATTEMPT {attempt_idx}/{max_attempts} evaluated."]
    lines.append(f"pipeline_ops: {pipeline_ops}")
    if error:
        lines.append(f"downstream_error: {error}")
    lines.append(f"downstream_fitness: {fit_str}")
    if metrics:
        lines.append(f"downstream_metrics: {metrics}")
    lines.append(f"best_fitness_so_far: {best_str}")
    if attempts_left <= 0:
        reminder = (
            "Solution-attempt budget exhausted. The system will keep the "
            "best-scoring attempt as the final pipeline."
        )
    else:
        reminder = (
            f"You have {attempts_left} more <solution> attempt(s). "
            "Submit a refined <solution>... --> Terminate</solution> to try a "
            "higher downstream_fitness, or repeat the current best to accept it."
        )
    lines.append("<reminder>" + reminder + "</reminder>")
    lines.append("</observation>")
    return "\n".join(lines)


__all__ = [
    "SYSTEM_TREE_SEARCH",
    "render_user_initial",
    "render_observation",
    "render_backtrack_hint",
    "render_solution_feedback",
]
