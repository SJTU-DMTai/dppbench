"""Prompt templates for the DataMaster agent.

Compared to DeepPrep:
* Single action tag ``<solution>...</solution>`` (one chain per black node;
  no exploration / backtracking turn).
* New tag ``<finding>...</finding>`` so the LLM can write a one-sentence
  takeaway that goes back into the GlobalMemory.
* The user prompt injects (1) the operator whitelist, (2) the current
  sandbox observation at the parent node, and (3) the GlobalMemory context
  (parent + siblings + global top-K).

The chain syntax is identical to DeepPrep: ``Op1(arg=value) --> Op2(...)
--> Terminate`` so we can reuse :func:`baselines.DeepPrep.tree_agent.parse_response`
and :func:`chain_to_steps`.
"""
from __future__ import annotations

from typing import Optional


SYSTEM_DATAMASTER = """You are DataMaster, a data-centric agent that
constructs preprocessing pipelines via tree-based search.

# Action protocol
Every reply MUST contain exactly ONE action tag:

  <solution>Op1(arg=value) --> Op2(...) --> ... --> Terminate</solution>

The `<solution>` chain you write is the DELTA pipeline that will be
applied on top of the parent node's accumulated pipeline. It MUST end with
`Terminate`. The system will:
  1. Concatenate (parent.accumulated_steps + your delta).
  2. Repair the pipeline structurally (mandatory ops, canonical category order).
  3. Execute the chain in a sandbox.
  4. Train the downstream model on a small subsample and record the
     resulting fitness as ``y_v`` for this black node.
  5. Backpropagate the reward up the tree (UCB statistics).

You may also include EXACTLY ONE optional reasoning tag and ONE optional
finding tag:

  <plan>brief plan for the remaining branch</plan>
  <finding>one sentence takeaway that other branches should know</finding>

The `<finding>` content is appended to the GlobalMemory of the new black
node, so future siblings/cousins can read it.

# Constraints
- You may ONLY use operators from the whitelist injected in the user message.
- You may NOT invent operator names or new arguments.
- You may NOT write Python code; the only way to act on the data is via
  the operator chain.
- Parameters that depend on the dataset (column lists, target columns, etc.)
  may be omitted; the system will fill them with safe defaults.
- A short, focused chain that improves on the parent's `fitness` is
  preferred over a long chain that repeats the parent's choices.
- Use the GlobalMemory section to AVOID repeating decisions that already
  failed on a sibling branch, and to BORROW configurations that worked
  globally.
"""


def render_user_initial(
    *,
    task_type: str,
    data_name: str,
    summary_text: str,
    op_descriptions: str,
    memory_context: str,
    parent_obs_text: str,
    parent_ops: list[str],
    parent_fitness: Optional[float],
    max_chain_len: int,
    expansion_idx: int,
    k_black: int,
    step: int,
    max_iterations: int,
    c_t: float,
    ordering_hint: str = "",
) -> str:
    target_text = (
        "downstream LightGBM (binary AUC)"
        if task_type == "tabular"
        else "downstream DIN recommender (AUC)"
    )
    fit_str = (
        f"{parent_fitness:.4f}" if isinstance(parent_fitness, float) else "n/a (root)"
    )
    parent_chain = " -> ".join(parent_ops) if parent_ops else "(empty pipeline; root)"
    return f"""# Task
Construct a data preparation pipeline for the **{task_type}** dataset
`{data_name}`. The dataset will eventually be consumed by a {target_text},
but the training itself is performed AFTER the pipeline is fixed and is
NOT something you have to implement.

# Dataset Summary
{summary_text}

# Search progress
- iteration: {step + 1} / {max_iterations}
- this expansion: {expansion_idx + 1} / {k_black} black-children of the chosen parent
- exploration coefficient c_t: {c_t:.3f}

# Parent Node State
- accumulated ops: {parent_chain}
- parent fitness:  {fit_str}

## Sandbox Observation (state AFTER applying the parent's accumulated ops)
{parent_obs_text}

# Global Memory
{memory_context}

# Operator Whitelist
Use ONLY these operators (grouped by category):
{op_descriptions}

# Canonical Category Order (HARD constraint)
Your accumulated pipeline MUST contain operators in non-decreasing
category-rank order. The system rejects out-of-order chains. Order is:
{ordering_hint}

# Limits
- Max delta-chain length: {max_chain_len}

Reason briefly inside <plan>...</plan>, OPTIONALLY emit one
<finding>...</finding> with a one-sentence takeaway, then issue exactly
ONE <solution>Op1(...) --> ... --> Terminate</solution> describing the
DELTA chain for this black child.
""".strip()


def render_retry_feedback(error: str, attempts_left: int, max_chain_len: int) -> str:
    return (
        "<observation>SOLUTION REJECTED.\n"
        f"reason: {error}\n"
        f"attempts_left for THIS black child: {attempts_left}\n"
        f"Submit a new <solution>... --> Terminate</solution> chain "
        f"(max length {max_chain_len}). Do not repeat the rejected ops/args."
        "</observation>"
    )


__all__ = [
    "SYSTEM_DATAMASTER",
    "render_user_initial",
    "render_retry_feedback",
]
