"""BAT prompts aligned with the official five-action DPAS protocol."""
from __future__ import annotations

from typing import Optional


SYSTEM_BAT = """You are BAT, a target-instance-free data-preparation
synthesizer driven by Monte Carlo Tree Search. Infer transformations from
source table samples and a target schema. DPPBench requires transformations
to be expressed only with operators from the supplied shared catalog; never
write arbitrary pandas or Python code."""


def render_schema_match(
    *, task_type: str, data_name: str, summary_text: str, obs_text: str,
    expected_target_columns: list[str],
) -> str:
    return f"""You are a meticulous schema matcher. Map every target column to
the source table columns that contribute to it. A target column may come from
one or several source tables or be derived by an operator.

Source Tables:
{obs_text}

Source Data Description:
{summary_text}

Target Table Columns:
{expected_target_columns}

First provide a short reasoning process, then return the final mapping as a
JSON list inside a ```json``` block. Each item must have `target_column` and
`sources`, where `sources` maps table names to lists of source columns.
"""


def render_identify_functions(
    *, task_type: str, data_name: str, summary_text: str, obs_text: str,
    schema_match_text: str, family_descriptions: str,
) -> str:
    return f"""Identify the data-preparation function families needed to
transform the source tables into the target schema. Follow the answer style of
listing each function family and explaining which columns/tables it applies to.

Source Tables:
{obs_text}

Target/schema mapping:
{schema_match_text}

Available function families and their DPPBench operators:
{family_descriptions}

Previous context:
{summary_text}
"""


def render_transformation(
    *, task_type: str, data_name: str, summary_text: str, obs_text: str,
    op_descriptions: str, schema_match_text: str,
    column_functions_text: str, max_chain_len: int,
) -> str:
    return f"""Generate the complete transformation for the following BAT task.
The original BAT emits pandas code; this DPPBench adaptation must instead emit
a concrete shared-operator chain. Do not omit required parameters and do not
invent operators. The chain must contain at most {max_chain_len} operators and
end with Terminate.

Source Tables:
{obs_text}

Dataset description:
{summary_text}

Schema match:
{schema_match_text}

Possible column functions:
{column_functions_text}

DPPBench operator catalog:
{op_descriptions}

Return only a JSON object inside ```json```:
```json
{{
  "chain_of_thought_reasoning": "step-by-step transformation reasoning",
  "pipeline": "Op1(arg=value) --> Op2(arg=value) --> Terminate"
}}
```
"""


def render_transformation_revision(
    *, task_type: str, data_name: str, op_descriptions: str,
    obs_after_exec: str, original_pipeline_ops: list[str],
    columns_match: Optional[bool], column_similarity: Optional[float],
    downstream_fitness: Optional[float], downstream_metrics: dict,
    exec_error: Optional[str], max_chain_len: int,
) -> str:
    sim = f"{column_similarity:.4f}" if isinstance(column_similarity, float) else "n/a"
    return f"""The previous BAT transformation failed at runtime or did not
exactly match the target columns. Analyze the execution result and return a
corrected complete operator chain.

Original pipeline: {original_pipeline_ops}
Execution result:
{obs_after_exec}
Execution error: {exec_error}
Exact columns match: {columns_match}
Target-column recall: {sim}

Operator catalog:
{op_descriptions}

The corrected chain must contain at most {max_chain_len} operators and end in
Terminate. Return only a JSON object inside ```json``` with keys
`chain_of_thought_reasoning` and `pipeline`, using the same format as the
Transformation action.
"""


def render_reward_judge(
    *, task_type: str, data_name: str, obs_after_exec: str,
    pipeline_ops: list[str], columns_match: Optional[bool],
    column_similarity: Optional[float], downstream_fitness: Optional[float],
    exec_error: Optional[str],
) -> str:
    return f"""Evaluate whether this transformation correctly produces the
target schema. Return a ```json``` object with `reason` and `reward`; reward
must be one of "1", "0.5", or "0".

Pipeline: {pipeline_ops}
Result: {obs_after_exec}
columns_match: {columns_match}
column_similarity: {column_similarity}
execution_error: {exec_error}
"""


__all__ = [
    "SYSTEM_BAT", "render_schema_match", "render_identify_functions",
    "render_transformation", "render_transformation_revision",
    "render_reward_judge",
]
