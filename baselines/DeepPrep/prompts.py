"""Prompts for DeepPrep's execution-grounded tree inference protocol."""
from __future__ import annotations

from typing import Optional


SYSTEM_TREE_SEARCH = """You are DeepPrep, an execution-grounded data-preparation agent.
You search a materialized tree of intermediate table states.

# Protocol
Every response MUST contain one reasoning block followed by exactly one action.
Use <plan>...</plan>. If your native chat template already starts a
<think>...</think> block, that block is accepted as the plan; do not emit both.

1. <expand>FULL ROOT-TO-NODE PATH --> OpA(...) --> OpB(...)</expand>
   Emit the complete path: copy an existing materialized root-to-node path,
   then append new concrete operator instances. For the root, emit only the
   new instances. Every copied parameter must match the environment exactly.

2. <answer>FULL ROOT-TO-LEAF PATH</answer>
   Return the complete path of an existing leaf whose materialized tables
   satisfy the target schema. Use <answer>root</answer> only when no operator
   is needed.

The environment, not you, emits <execute>...</execute> after an expansion.
Never emit <execute>, <operator>, <solution>, or <backtrack> yourself.

# Constraints
- Use only operators and parameter values in the injected DPPBench catalog.
- Every operator instance must be concrete and use keyword arguments only.
- Required parameters may not be omitted. The environment never guesses or
  randomly fills missing parameters.
- Categories are descriptive and do not impose an execution order.
- To abandon a branch, choose another materialized parent in the next <expand>.
- Do not claim success until the current tables satisfy the target schema.
"""


def render_user_initial(
    task_type: str,
    data_name: str,
    summary_text: str,
    required_ops_text: str,
    target_schema_text: str,
    op_descriptions: str,
    obs_text: str,
    max_chain_len: int,
    max_explore_turn: int,
) -> str:
    return f"""# Task
Construct an executable data-preparation pipeline for the **{task_type}**
dataset `{data_name}`.

# Source-table summary
{summary_text}

# Mandatory DPPBench contract
The final answered path MUST contain every operator listed below. These are
context-dependent benchmark requirements, not suggestions, and the environment
will reject an answer that omits any of them.
{required_ops_text}

# Target schema Sigma*
{target_schema_text}

# Root materialized state
{obs_text}

# DPPBench operator catalog
{op_descriptions}

# Limits
- Maximum operators in one <expand>: {max_chain_len}
- Maximum reasoning turns: {max_explore_turn}

Emit one <plan> (or the native <think> block already started by your chat
template), then exactly one <expand> or <answer> action.
""".strip()


def render_execute(
    *,
    parent_path: str,
    executed_path: str,
    obs_text: str,
    error: Optional[str],
    turn_left: int,
) -> str:
    status = "FAILED" if error else "OK"
    lines = [f'<execute status="{status}" parent="{parent_path}">']
    if executed_path:
        lines.append(f"materialized_path: {executed_path}")
    if error:
        lines.append(f"error: {error}")
        lines.append("No child was created for the failed operator.")
    elif obs_text:
        lines.append(obs_text)
    lines.append(f"turns_left: {max(0, turn_left)}")
    lines.append("</execute>")
    return "\n".join(lines)


def render_protocol_error(error: str, turn_left: int) -> str:
    return (
        '<execute status="FAILED" parent="unchanged">\n'
        f"error: {error}\n"
        "No tree state was changed. Emit one <plan> (or native <think>) "
        "followed by exactly one <expand> or <answer>.\n"
        f"turns_left: {max(0, turn_left)}\n"
        "</execute>"
    )


__all__ = [
    "SYSTEM_TREE_SEARCH",
    "render_user_initial",
    "render_execute",
    "render_protocol_error",
]
