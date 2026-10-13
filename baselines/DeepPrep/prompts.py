"""Prompts for DeepPrep's execution-grounded tree inference protocol."""
from __future__ import annotations

from typing import Optional


# Prefer the released checkpoint's native action format. Mixing its native
# closing tags with a newly introduced <plan> opening caused plan-only outputs.
SYSTEM_TREE_SEARCH = """You are DeepPrep, an execution-grounded data-preparation agent.
Search the materialized tree using only the supplied DPPBench operators.
Output a brief explanation (at most 80 words), then a concrete FULL ROOT-TO-NODE
operator chain on its own line, ending with </execute>. Example:
Clean missing values and encode categorical features.
HandleMV(method='median', action='impute') --> LabelEncode(cols=None) </execute>
Do not emit <plan>, Markdown, catalog entries, or invented execution results.
The environment executes your chain and reports the materialized path.
To finish, return <answer>EXACT EXISTING MATERIALIZED PATH</answer>.
The <expand>CHAIN</expand> format is also accepted. Never return reasoning alone.
Copy an existing path exactly before appending operators; do not invent a parent.
Every mandatory operator must appear, even if you consider it unnecessary.
Do not reinterpret required_tables or mandatory operators as optional.
Use exact case-sensitive names and concrete calls, never HANDLEMV --> CASTTYPE
or operator names with explanatory parentheses. Extra joined columns are expected,
not an error. Integer user/item IDs need not be cast to strings.
CastType requires a dictionary, e.g. CastType(col_dtypes={'rating': 'float64'}).
Use keyword arguments only. The environment supplies input tables; never pass
positional table arguments, df=, or table= to common operators.
Legal syntax examples (use actual column/table names from context):
SortRows(by=['timestamp'], ascending=True)
JoinTable(method='rec', item_df='$item_df', item_col='item_id')
For key/agg joins use aux_df='$TABLE_df' and key_col='COLUMN'; do not use item_df/user_df.
Two input nodes for JoinTable means two tables, not two preceding transformations.
To extend an existing path, repeat that full path before --> JoinTable(...).
Write the chain on one line; do not number operators or put explanations inside parentheses.
CreateSequence(user_col='user_id', item_col='item_id', time_col='timestamp', seq_col='item_id_seq', max_len=50)
CreateSequence uses seq_col and max_len, not sequence_col, max_sequence_length,
or target_col. LabelEncode(cols=None) automatically selects categorical columns;
prefer this to enumerating hundreds of columns. Do not encode numeric V* columns
merely because they exist. Preserve target, identifier, and __split__ columns.
Examples illustrate syntax, not a fixed solution. Never repeat reasoning or tags.
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

Emit a brief explanation, then the concrete chain ending in </execute>.
Alternatively emit exactly one <expand> or <answer> action. Never emit only a plan.
""".strip()


def render_execute(
    *,
    parent_path: str,
    executed_path: str,
    obs_text: str,
    error: Optional[str],
    turn_left: int,
    remaining_required_ops: Optional[list[str]] = None,
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
    if remaining_required_ops is not None:
        lines.append(f"mandatory_ops_remaining: {remaining_required_ops}")
    lines.append(f"turns_left: {max(0, turn_left)}")
    lines.append("</execute>")
    return "\n".join(lines)


def render_protocol_error(error: str, turn_left: int) -> str:
    return (
        '<execute status="FAILED" parent="unchanged">\n'
        f"error: {error}\n"
        "No tree state was changed. Output a concrete chain ending in </execute>, "
        "or one <expand> or <answer> action. Reasoning alone is not an action.\n"
        "Keep the plan under 80 words; do not repeat previous reasoning.\n"
        "Syntax example: HandleMV(method='median', action='impute') </execute>\n"
        f"turns_left: {max(0, turn_left)}\n"
        "</execute>"
    )


__all__ = [
    "SYSTEM_TREE_SEARCH",
    "render_user_initial",
    "render_execute",
    "render_protocol_error",
]
