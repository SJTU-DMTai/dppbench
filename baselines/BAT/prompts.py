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
    if len(expected_target_columns) > 40:
        return f"""Match target columns to source columns for {data_name}.
Direct same-name columns are implicit identity matches. List only target
columns that are missing from source tables or need derivation. If all target
columns are direct matches, return an empty JSON list. Return JSON only;
do not provide reasoning or enumerate direct matches.

Source Tables:
{obs_text}

Target rule and task context:
{summary_text}

Return a JSON list inside ```json```. Each item must have `target_column`
and `sources`, where `sources` maps table names to source columns.
"""
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
Use at most six short bullets. Refer to column groups (numeric, categorical,
datetime) rather than enumerating long lists of column names.

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
For tabular tasks use LabelEncode(cols=None) to cover all object columns;
partial hand-written lists can leave nonnumeric model inputs. Prefer
HandleMV(cols=None, method='median', action='impute') for numeric imputation.
Only use names and keywords in the catalog. Preserve list-valued sequence
features, and never transform the internal __split__ column.
Every operator listed under mandatory_operators in the dataset description
must appear in the chain. JoinTable is mandatory whenever a secondary source
exists, even if its columns are absent from the target schema; the DAG must
connect every source to train.

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
  "pipeline": "Op1(arg=value) --> Op2(arg=value) --> Terminate",
  "chain_of_thought_reasoning": "one sentence justifying task-specific operators"
}}
```
"""


def render_transformation_revision(
    *, task_type: str, data_name: str, op_descriptions: str,
    summary_text: str,
    obs_after_exec: str, original_pipeline_ops: list[str],
    columns_match: Optional[bool], column_similarity: Optional[float],
    downstream_fitness: Optional[float], downstream_metrics: dict,
    exec_error: Optional[str], max_chain_len: int,
    downstream_error: Optional[str] = None,
) -> str:
    sim = f"{column_similarity:.4f}" if isinstance(column_similarity, float) else "n/a"
    return f"""Review the previous BAT transformation using the actual feedback
below. Correct reported execution or training errors. If errors are absent,
propose a task-relevant alternative without assuming existing features are wrong.
Return a complete operator chain using only catalog names and keywords.

Original pipeline: {original_pipeline_ops}
Dataset description and required operator examples:
{summary_text}
Execution result:
{obs_after_exec}
Execution error: {exec_error}
Downstream validation-training error: {downstream_error}
Validation AUC: {downstream_fitness}
Validation metrics: {downstream_metrics}
Fix the downstream error as well as schema problems. Encode all object columns
with LabelEncode(cols=None) when supported; do not enumerate hundreds of names.
For object-dtype errors, LabelEncode(cols=None) already returns numeric values:
do not invent a casting operator. Preserve list-valued sequence features;
never encode sequence lists or the internal __split__ column.
Exact columns match: {columns_match}
Target-column recall: {sim}
Every operator listed under mandatory_operators below must appear in the
corrected chain. A secondary source requires JoinTable even when its columns
are not target columns.

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
