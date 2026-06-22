"""Prompt templates for the SPIO baseline.

SPIO performs two LLM-driven phases:

  1. **NL plan**: a single call asks the model to draft a strategy for each
     of the 5 stages (integration / cleaning / preprocessing /
     feature_engineering / transformation). The output is a structured
     section per stage.
  2. **Per-stage code generation**: for each of the 4 *single-table*
     stages, the model is asked to author N candidate ``CustomOp``
     snippets that strictly satisfy the contract documented in
     ``CONTRACT_SPEC``.

All snippet replies must be wrapped in a ```python ... ``` fence so the
parser in :func:`parse_code_block` can extract them robustly.
"""
from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Contract specification (kept in one place; injected into both prompts)
# ---------------------------------------------------------------------------
CONTRACT_SPEC = """\
**CustomOp code contract — the snippet you author MUST satisfy ALL of:**
- Define exactly one entry function:
      def pipeline(df):
          ...
          return df
- Single ``pandas.DataFrame`` in, single ``pandas.DataFrame`` out.
- Do **not** modify ``df`` in place; copy it first (``df = df.copy()``).
- **No** ``import`` / ``from ... import`` statements anywhere. The
  following names are pre-injected and may be used directly:
      pd  -> pandas
      np  -> numpy
      re  -> re
      math -> math
      preprocessing -> sklearn.preprocessing
      impute        -> sklearn.impute
- **No** I/O, **no** ``print``, **no** ``open``, **no** ``eval`` /
  ``exec`` / ``__import__`` / ``getattr`` / ``setattr``, **no** dunder
  attribute access (``__class__`` etc.).
- Reference only the passed ``df`` argument — do not depend on globals,
  other tables, or random state outside ``np.random`` / ``pd``.
- Be defensive: guard column accesses with ``if col in df.columns``,
  cast safely, fillna before scaling, etc. ``df`` may already have been
  partially transformed by previous stages.
- Be split-safe for tabular tasks: this snippet is applied independently
  to ``train_df`` and ``test_df``, so prefer stateless transformations
  (e.g. fit & transform within the same call). Do not leak target
  statistics across rows.
"""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_SPIO = (
    "You are SPIO (Sequential Plan Integration and Optimization), an LLM "
    "agent that designs a multi-stage data preparation pipeline by\n"
    "  (a) drafting a per-stage strategy in natural language, and\n"
    "  (b) authoring concrete Python snippets for each single-table stage.\n"
    "Every snippet you produce MUST follow the CustomOp contract — the\n"
    "execution sandbox rejects imports, I/O, dunder access and any function\n"
    "other than ``def pipeline(df): ... return df``.\n\n"
    + CONTRACT_SPEC
)


# ---------------------------------------------------------------------------
# Plan prompt (one shot, covers all 5 stages)
# ---------------------------------------------------------------------------
def render_plan_prompt(
    *,
    task_type: str,
    data_name: str,
    summary_text: str,
    obs_text: str,
    stage_hints: dict[str, str],
) -> str:
    target = (
        "downstream LightGBM (binary AUC) on the validation split"
        if task_type == "tabular"
        else "downstream DIN recommender (AUC) on the held-out eval split"
    )
    hint_blocks = []
    for stage in (
        "integration",
        "cleaning",
        "preprocessing",
        "feature_engineering",
        "transformation",
    ):
        block = stage_hints.get(stage, "(no typical operators)")
        hint_blocks.append(f"### {stage}\n{block}")
    hints_md = "\n\n".join(hint_blocks)

    return f"""# Task
Plan a 5-stage data preparation pipeline for the **{task_type}** dataset
`{data_name}`. The pipeline output is consumed by a {target}.

# Dataset summary
{summary_text}

## Initial sandbox observation
{obs_text}

# Stages and typical operators
For each stage, below are the categories / example operators that the
dppbench framework typically provides. You are not required to use these
operators (you will instead author Python snippets in a later step), but
the stage planning should stay within each stage's responsibility.

{hints_md}

# What to produce now
Output a *natural-language* plan with one section per stage, following
this exact template (Markdown headings + bullet points):

## Stage 1: integration
- ...

## Stage 2: cleaning
- ...

## Stage 3: preprocessing
- ...

## Stage 4: feature_engineering
- ...

## Stage 5: transformation
- ...

Keep each section concise (2-5 short bullets). Do **not** write code in
this reply — code generation happens in subsequent turns. Focus on what
the stage should accomplish given the dataset above.
""".strip()


# ---------------------------------------------------------------------------
# Per-stage code generation prompt
# ---------------------------------------------------------------------------
def render_codegen_prompt(
    *,
    task_type: str,
    data_name: str,
    stage: str,
    stage_plan: str,
    summary_text: str,
    obs_text: str,
    n_candidates: int,
) -> str:
    target = (
        "downstream LightGBM (binary AUC)"
        if task_type == "tabular"
        else "downstream DIN recommender (AUC)"
    )
    target_table = (
        "train_df / test_df (independently applied)"
        if task_type == "tabular"
        else "interaction_df"
    )
    return f"""# Cascade context
Dataset: **{data_name}** (task_type={task_type})
Optimisation target: {target}
This snippet operates on: {target_table}

# Dataset summary (initial)
{summary_text}

# Current data state (after the previously selected stages)
{obs_text}

# Current stage: {stage}
The natural-language plan for this stage was:
{stage_plan.strip()}

# Your task
Produce **{n_candidates} alternative** Python snippets that implement the
``{stage}`` stage as a single ``CustomOp`` function. Each snippet is a
complete standalone implementation of ``def pipeline(df): ... return df``.

The candidates should differ in approach (e.g. different imputation /
encoding / scaling / feature-creation choices) so the search can pick the
best one against the validation metric.

{CONTRACT_SPEC}

# Output format (STRICT)
Reply with EXACTLY {n_candidates} fenced Python blocks, one per
candidate, in this order:

```python
# Candidate 1
def pipeline(df):
    df = df.copy()
    # ... your transform here ...
    return df
```

```python
# Candidate 2
def pipeline(df):
    df = df.copy()
    # ... your transform here ...
    return df
```

No prose between or after the blocks. Each block must define
``def pipeline(df)`` and only ``def pipeline(df)``.
""".strip()


# ---------------------------------------------------------------------------
# Output parsing helpers
# ---------------------------------------------------------------------------
_FENCED_PY_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE
)

_STAGE_HEADER_RE = re.compile(
    r"^\s*##\s*Stage\s*(\d)\s*:\s*([a-z_]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_code_blocks(text: str) -> list[str]:
    """Extract every ```python ... ``` (or unlabelled) fenced block.

    Returns the list of inner code strings (stripped). If no fences are
    found, attempts a best-effort fallback: if the whole reply is a
    function definition, return it as a single block.
    """
    blocks = [m.group(1).strip() for m in _FENCED_PY_RE.finditer(text)]
    if blocks:
        return blocks
    stripped = text.strip()
    if stripped.startswith("def pipeline"):
        return [stripped]
    return []


def parse_plan_sections(text: str) -> dict[str, str]:
    """Split an NL-plan reply into ``{stage_name: section_text}``.

    Recognises the ``## Stage <n>: <stage>`` headers from
    :func:`render_plan_prompt`. Stage names are normalised to lowercase
    with the variants ``feature engineering``/``feature-engineering`` /
    ``feature_engineering`` all mapping to ``feature_engineering``.
    """
    sections: dict[str, str] = {}
    matches = list(_STAGE_HEADER_RE.finditer(text))
    if not matches:
        return sections
    for i, m in enumerate(matches):
        raw_stage = m.group(2).lower().strip()
        norm = (
            raw_stage.replace(" ", "_")
            .replace("-", "_")
        )
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[norm] = text[start:end].strip()
    return sections


__all__ = [
    "SYSTEM_SPIO",
    "CONTRACT_SPEC",
    "render_plan_prompt",
    "render_codegen_prompt",
    "parse_code_blocks",
    "parse_plan_sections",
]
