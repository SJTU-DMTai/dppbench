"""Prompt templates for the BAT baseline.

The five prompts mirror BAT's original ``get_prompt.py`` structure
(SchemaMatch / IdentifyColumnFunctions / Transformation / Revision /
Reward) but are adapted to dppbench: instead of asking the LLM to write
free-form pandas code, BAT asks the LLM to compose an operator chain
``Op1(arg=val) --> Op2 --> ... --> Terminate`` from the shared dppbench 58-op
whitelist, parsable by ``baselines.DeepPrep.tree_agent.chain_to_steps``.
"""
from __future__ import annotations

from typing import Optional


SYSTEM_BAT = """You are BAT, a target-instance-free data-preparation
synthesizer driven by Monte Carlo Tree Search.

You operate inside dppbench. Your output is consumed in five distinct
roles, one per node type:
  - SchemaMatch          -> JSON column-correspondence proposal
  - IdentifyColumnFunctions -> JSON function-family proposal per column
  - Transformation       -> a `<pipeline>...</pipeline>` operator chain
  - TransformationRevision -> a revised `<pipeline>...</pipeline>` chain
  - RewardJudge          -> integer score 0/0.5/1

Strict rules:
  * Only use operator names from the injected whitelist; never invent
    operator names or write raw pandas code.
  * Always end the operator chain with `Terminate`.
  * Parameters that depend on the dataset (column lists, target column,
    auxiliary tables) may be omitted; the system fills sensible defaults
    based on the inferred ``DataContext``.
  * BAT scores each terminal pipeline against (i) target-instance-free
    schema similarity, (ii) optional downstream-model fitness on a small
    subsample, (iii) optional self-judgement.
"""


# ---------------------------------------------------------------------------
# Schema match
# ---------------------------------------------------------------------------
def render_schema_match(
    *,
    task_type: str,
    data_name: str,
    summary_text: str,
    obs_text: str,
    expected_target_columns: list[str],
) -> str:
    return f"""# SchemaMatch (BAT node type: SCHEMA_MATCH)
Dataset: **{data_name}** ({task_type})

## Source schema (sandbox observation)
{obs_text}

## Dataset summary
{summary_text}

## Expected post-pipeline columns (target-instance-free)
{expected_target_columns}

## Output protocol
Return a JSON object whose keys are expected target columns and whose
values are lists of source columns (or aux-table column references) that
should map to that target. Wrap the JSON in <schema_match>...</schema_match>.
Example:

<schema_match>
{{
  "user_id": ["user_id"],
  "label":   ["click", "is_click"],
  "item_id_seq": ["item_id"]
}}
</schema_match>
"""


# ---------------------------------------------------------------------------
# Identify column functions
# ---------------------------------------------------------------------------
def render_identify_functions(
    *,
    task_type: str,
    data_name: str,
    summary_text: str,
    obs_text: str,
    schema_match_text: str,
    family_descriptions: str,
) -> str:
    return f"""# IdentifyColumnFunctions (BAT node type: IDENTIFY_COLUMN_FUNCTIONS)
Dataset: **{data_name}** ({task_type})

## Sandbox observation
{obs_text}

## Dataset summary
{summary_text}

## Prior schema match (may be empty)
{schema_match_text}

## Available function families (mapped to dppbench OpCategories)
{family_descriptions}

## Output protocol
Pick, for each interesting column, the function family that should
transform it. Wrap the answer in
<column_functions>...</column_functions> as JSON. Example:

<column_functions>
{{
  "amount":     "scale_or_normalize",
  "timestamp":  "datetime_format",
  "category":   "encoding",
  "user_id":    "join"
}}
</column_functions>
"""


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------
def render_transformation(
    *,
    task_type: str,
    data_name: str,
    summary_text: str,
    obs_text: str,
    op_descriptions: str,
    schema_match_text: str,
    column_functions_text: str,
    max_chain_len: int,
) -> str:
    return f"""# Transformation (BAT node type: TRANSFORMATION)
Dataset: **{data_name}** ({task_type})

## Sandbox observation
{obs_text}

## Dataset summary
{summary_text}

## Prior schema match (may be empty)
{schema_match_text}

## Prior column functions (may be empty)
{column_functions_text}

## Operator whitelist (use ONLY these)
{op_descriptions}

## Output protocol
Compose an operator chain that transforms the source data into a form
ready for the downstream model. Wrap it inside
<pipeline>...</pipeline>. Length budget: at most {max_chain_len} ops.
The chain MUST end with `Terminate`. Example:

<pipeline>
JoinTable --> CreateSequence(seq_len=20) --> SampleNegative(n_negatives=1) --> Terminate
</pipeline>
"""


# ---------------------------------------------------------------------------
# Transformation revision (post-execution)
# ---------------------------------------------------------------------------
def render_transformation_revision(
    *,
    task_type: str,
    data_name: str,
    op_descriptions: str,
    obs_after_exec: str,
    original_pipeline_ops: list[str],
    columns_match: Optional[bool],
    column_similarity: Optional[float],
    downstream_fitness: Optional[float],
    downstream_metrics: dict,
    exec_error: Optional[str],
    max_chain_len: int,
) -> str:
    sim_str = (f"{column_similarity:.4f}"
               if isinstance(column_similarity, float) else "n/a")
    fit_str = (f"{downstream_fitness:.4f}"
               if isinstance(downstream_fitness, float) else "n/a")
    return f"""# TransformationRevision (BAT node type: REVISED_TRANSFORMATION)
Dataset: **{data_name}** ({task_type})

The previously emitted pipeline did not satisfy the BAT acceptance
criteria. Revise it.

## Original operator chain
{original_pipeline_ops}

## Observation after the failed execution (or last successful state)
{obs_after_exec}

## Feedback
- columns_match: {columns_match}
- column_similarity: {sim_str}
- downstream_fitness (small-N): {fit_str}
- downstream_metrics: {downstream_metrics}
- exec_error: {exec_error}

## Operator whitelist (use ONLY these)
{op_descriptions}

## Output protocol
Wrap a corrected operator chain inside <pipeline>...</pipeline>, ending
with `Terminate`. Length budget: {max_chain_len} ops.
"""


# ---------------------------------------------------------------------------
# Reward judge (LLM scoring fallback)
# ---------------------------------------------------------------------------
def render_reward_judge(
    *,
    task_type: str,
    data_name: str,
    obs_after_exec: str,
    pipeline_ops: list[str],
    columns_match: Optional[bool],
    column_similarity: Optional[float],
    downstream_fitness: Optional[float],
    exec_error: Optional[str],
) -> str:
    sim_str = (f"{column_similarity:.4f}"
               if isinstance(column_similarity, float) else "n/a")
    fit_str = (f"{downstream_fitness:.4f}"
               if isinstance(downstream_fitness, float) else "n/a")
    return f"""# RewardJudge (BAT node type: END)
Dataset: **{data_name}** ({task_type})

## Operator chain
{pipeline_ops}

## Sandbox observation after execution
{obs_after_exec}

## Computed signals
- columns_match: {columns_match}
- column_similarity: {sim_str}
- downstream_fitness (small-N): {fit_str}
- exec_error: {exec_error}

## Output protocol
Reply with a single floating-point reward in {{0, 0.5, 1}} wrapped in
<reward>...</reward>. Use 1 if the pipeline both reproduces the expected
schema and yields a healthy downstream metric; 0.5 if partial; 0 if
broken.
"""


__all__ = [
    "SYSTEM_BAT",
    "render_schema_match",
    "render_identify_functions",
    "render_transformation",
    "render_transformation_revision",
    "render_reward_judge",
]
