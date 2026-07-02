"""BAT MCTS actions.

Each :class:`BaseAction` knows how to materialise children of a given
parent node by querying the LLM. The action set faithfully reproduces
BAT's DPAS:

  * SchemaMatch
  * IdentifyColumnFunctions
  * Transformation
  * TransformationRevision
  * End

The legality table :func:`get_valid_action_space_for_node` mirrors
``ZJU-DAILY/BAT/src/mcts/node.py`` (with the additional rule that an
action class cannot reappear on the same path).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from baselines.DeepPrep.tree_agent import ChainParseError, chain_to_steps
from baselines.common.pipeline import DataContext, Pipeline, PipelineStep
from baselines.common.pipeline_constraints import is_legal, repair

from . import prompts
from .node import MCTSNode, MCTSNodeType
from .operator_catalog import (
    format_op_descriptions,
    function_family_descriptions,
)

if TYPE_CHECKING:
    from .sandbox import Sandbox
    from baselines.DeepPrep.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_PIPE_RE = re.compile(r"<pipeline>(.*?)</pipeline>", re.DOTALL)
_SCHEMA_MATCH_RE = re.compile(r"<schema_match>(.*?)</schema_match>", re.DOTALL)
_COL_FUNC_RE = re.compile(r"<column_functions>(.*?)</column_functions>",
                           re.DOTALL)


def _extract(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text or "")
    return m.group(1).strip() if m else None


def _safe_json_loads(s: Optional[str]) -> Optional[dict]:
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        return None


def _summary_text(ctx: DataContext) -> str:
    return (
        f"task_type={ctx.task_type} dataset={ctx.data_name}\n"
        f"target_col={ctx.target_col} time_col={ctx.time_col} "
        f"id_col={ctx.id_col}\n"
        f"#numeric_cols={len(ctx.numeric_cols)} "
        f"#categorical_cols={len(ctx.categorical_cols)} "
        f"#list_cols={len(ctx.list_cols)} #text_cols={len(ctx.text_cols)}\n"
        f"has_user_df={ctx.has_user_df} has_item_df={ctx.has_item_df} "
        f"aux_dfs={ctx.aux_dfs}"
    )


def expected_target_columns(ctx: DataContext) -> list[str]:
    """Compute the BAT target-instance-free schema expectation."""
    cols: list[str] = []
    if ctx.task_type == "tabular":
        cols.extend(ctx.numeric_cols)
        cols.extend(ctx.categorical_cols)
        if ctx.target_col:
            cols.append(ctx.target_col)
    else:
        if ctx.user_col:
            cols.append(ctx.user_col)
        if ctx.item_col:
            cols.append(ctx.item_col)
        if ctx.target_col:
            cols.append(ctx.target_col)
        cols.extend(ctx.list_cols)
        # CreateSequence often emits "<col>_seq" alongside the originals.
        cols.append("item_id_seq")
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _sandbox_columns(sandbox: "Sandbox") -> list[str]:
    if sandbox.data is None:
        return []
    if sandbox.task_type == "tabular":
        df = getattr(sandbox.data, "train_df", None)
    else:
        df = getattr(sandbox.data, "interaction_df", None)
    if df is None:
        return []
    return [str(c) for c in df.columns]


def _column_similarity(actual: list[str], expected: list[str]) -> float:
    if not expected and not actual:
        return 1.0
    a = set(actual)
    e = set(expected)
    if not (a or e):
        return 1.0
    inter = a & e
    union = a | e
    return float(len(inter)) / float(len(union)) if union else 0.0


# ---------------------------------------------------------------------------
# Action base
# ---------------------------------------------------------------------------
@dataclass
class ActionContext:
    """Bag of references that every action needs."""
    llm: "LLMClient"
    ctx: DataContext
    sandbox: "Sandbox"
    rng: random.Random
    max_chain_len: int = 6
    verbose: bool = False


class BaseAction:
    """Base class for the five BAT actions."""

    name: str = "BASE"
    target_node_type: MCTSNodeType = MCTSNodeType.ROOT

    def create_children_nodes(
        self,
        parent: MCTSNode,
        action_ctx: ActionContext,
    ) -> list[MCTSNode]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Schema match
# ---------------------------------------------------------------------------
class SchemaMatchAction(BaseAction):
    name = "SchemaMatch"
    target_node_type = MCTSNodeType.SCHEMA_MATCH

    def create_children_nodes(self, parent, action_ctx):
        ctx = action_ctx.ctx
        sandbox = action_ctx.sandbox
        # Replay parent state for an up-to-date observation.
        obs = sandbox._observe()  # type: ignore[attr-defined]
        prompt = prompts.render_schema_match(
            task_type=ctx.task_type,
            data_name=ctx.data_name,
            summary_text=_summary_text(ctx),
            obs_text=obs.text,
            expected_target_columns=expected_target_columns(ctx),
        )
        try:
            response = action_ctx.llm.chat(
                [
                    {"role": "system", "content": prompts.SYSTEM_BAT},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as e:
            response = f"<schema_match>{{}}</schema_match>  (llm_error: {e})"
        sm_json = _safe_json_loads(_extract(_SCHEMA_MATCH_RE, response)) or {}
        child = MCTSNode(
            node_type=MCTSNodeType.SCHEMA_MATCH,
            parent=parent,
            parent_action=self,
            depth=parent.depth + 1,
            schema_match=sm_json,
        )
        return [child]


# ---------------------------------------------------------------------------
# Identify column functions
# ---------------------------------------------------------------------------
class IdentifyColumnFunctionsAction(BaseAction):
    name = "IdentifyColumnFunctions"
    target_node_type = MCTSNodeType.IDENTIFY_COLUMN_FUNCTIONS

    def create_children_nodes(self, parent, action_ctx):
        ctx = action_ctx.ctx
        sandbox = action_ctx.sandbox
        obs = sandbox._observe()  # type: ignore[attr-defined]
        # Inherit prior schema_match from the path if present.
        schema_match_text = "(none)"
        for node in parent.path_to_root():
            if node.schema_match:
                schema_match_text = json.dumps(node.schema_match, indent=2)
                break
        prompt = prompts.render_identify_functions(
            task_type=ctx.task_type,
            data_name=ctx.data_name,
            summary_text=_summary_text(ctx),
            obs_text=obs.text,
            schema_match_text=schema_match_text,
            family_descriptions=function_family_descriptions(ctx.task_type),
        )
        try:
            response = action_ctx.llm.chat(
                [
                    {"role": "system", "content": prompts.SYSTEM_BAT},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as e:
            response = f"<column_functions>{{}}</column_functions>  (llm_error: {e})"
        cf_json = _safe_json_loads(_extract(_COL_FUNC_RE, response)) or {}
        child = MCTSNode(
            node_type=MCTSNodeType.IDENTIFY_COLUMN_FUNCTIONS,
            parent=parent,
            parent_action=self,
            depth=parent.depth + 1,
            column_functions=json.dumps(cf_json),
        )
        return [child]


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------
class TransformationAction(BaseAction):
    name = "Transformation"
    target_node_type = MCTSNodeType.TRANSFORMATION

    def create_children_nodes(self, parent, action_ctx):
        ctx = action_ctx.ctx
        sandbox = action_ctx.sandbox
        obs = sandbox._observe()  # type: ignore[attr-defined]
        # Inherit schema_match / column_functions from the path if present.
        schema_match_text = "(none)"
        col_func_text = "(none)"
        for node in parent.path_to_root():
            if node.schema_match and schema_match_text == "(none)":
                schema_match_text = json.dumps(node.schema_match, indent=2)
            if node.column_functions and col_func_text == "(none)":
                col_func_text = node.column_functions
        prompt = prompts.render_transformation(
            task_type=ctx.task_type,
            data_name=ctx.data_name,
            summary_text=_summary_text(ctx),
            obs_text=obs.text,
            op_descriptions=format_op_descriptions(ctx.task_type),
            schema_match_text=schema_match_text,
            column_functions_text=col_func_text,
            max_chain_len=action_ctx.max_chain_len,
        )
        try:
            response = action_ctx.llm.chat(
                [
                    {"role": "system", "content": prompts.SYSTEM_BAT},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as e:
            response = f"<pipeline>Terminate</pipeline>  (llm_error: {e})"
        chain_str = _extract(_PIPE_RE, response) or ""
        steps, parse_error = _parse_chain_safe(chain_str, ctx, action_ctx.rng)
        child = MCTSNode(
            node_type=MCTSNodeType.TRANSFORMATION,
            parent=parent,
            parent_action=self,
            depth=parent.depth + 1,
            pipeline_steps=steps,
            exec_error=parse_error,
        )
        return [child]


# ---------------------------------------------------------------------------
# Transformation revision
# ---------------------------------------------------------------------------
class TransformationRevisionAction(BaseAction):
    name = "TransformationRevision"
    target_node_type = MCTSNodeType.REVISED_TRANSFORMATION

    def create_children_nodes(self, parent, action_ctx):
        ctx = action_ctx.ctx
        sandbox = action_ctx.sandbox
        obs = sandbox._observe()  # type: ignore[attr-defined]
        original_ops = [s.op for s in parent.pipeline_steps] or []
        prompt = prompts.render_transformation_revision(
            task_type=ctx.task_type,
            data_name=ctx.data_name,
            op_descriptions=format_op_descriptions(ctx.task_type),
            obs_after_exec=obs.text,
            original_pipeline_ops=original_ops,
            columns_match=parent.columns_match,
            column_similarity=parent.column_similarity,
            downstream_fitness=parent.downstream_fitness,
            downstream_metrics=parent.downstream_metrics,
            exec_error=parent.exec_error,
            max_chain_len=action_ctx.max_chain_len,
        )
        try:
            response = action_ctx.llm.chat(
                [
                    {"role": "system", "content": prompts.SYSTEM_BAT},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as e:
            response = f"<pipeline>Terminate</pipeline>  (llm_error: {e})"
        chain_str = _extract(_PIPE_RE, response) or ""
        steps, parse_error = _parse_chain_safe(chain_str, ctx, action_ctx.rng)
        child = MCTSNode(
            node_type=MCTSNodeType.REVISED_TRANSFORMATION,
            parent=parent,
            parent_action=self,
            depth=parent.depth + 1,
            revised_pipeline_steps=steps,
            exec_error=parse_error,
        )
        return [child]


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
class EndAction(BaseAction):
    name = "End"
    target_node_type = MCTSNodeType.END

    def create_children_nodes(self, parent, action_ctx):
        # Materialise the final pipeline by walking up the path and taking
        # the most recent operator chain (revision wins).
        steps = parent.latest_pipeline_steps()
        # Final structural repair so mandatory ops are always present.
        pipe = Pipeline(steps=list(steps))
        repair(pipe, action_ctx.ctx.task_type, action_ctx.ctx)
        child = MCTSNode(
            node_type=MCTSNodeType.END,
            parent=parent,
            parent_action=self,
            depth=parent.depth + 1,
            final_pipeline_steps=list(pipe.steps),
        )
        return [child]


# ---------------------------------------------------------------------------
# Chain parsing wrapper
# ---------------------------------------------------------------------------
def _parse_chain_safe(
    chain_str: str,
    ctx: DataContext,
    rng: random.Random,
) -> tuple[list[PipelineStep], Optional[str]]:
    if not chain_str.strip():
        return [], "empty pipeline chain"
    try:
        steps = chain_to_steps(chain_str, ctx, rng)
    except ChainParseError as e:
        return [], f"ChainParseError: {e}"
    except Exception as e:  # pragma: no cover - defensive
        return [], f"{type(e).__name__}: {e}"
    return steps, None


# ---------------------------------------------------------------------------
# Legality table -- mirrors ZJU-DAILY/BAT/src/mcts/node.py
# ---------------------------------------------------------------------------
_NODE_TO_ACTION_CLASSES: dict[MCTSNodeType, list[type[BaseAction]]] = {
    MCTSNodeType.ROOT: [
        SchemaMatchAction,
        IdentifyColumnFunctionsAction,
        TransformationAction,
    ],
    MCTSNodeType.SCHEMA_MATCH: [
        IdentifyColumnFunctionsAction,
        TransformationAction,
    ],
    MCTSNodeType.IDENTIFY_COLUMN_FUNCTIONS: [
        SchemaMatchAction,
        TransformationAction,
    ],
    MCTSNodeType.TRANSFORMATION: [
        EndAction,
        TransformationRevisionAction,
    ],
    MCTSNodeType.REVISED_TRANSFORMATION: [
        EndAction,
    ],
    MCTSNodeType.END: [],
}


def get_valid_action_space_for_node(node: MCTSNode) -> list[type[BaseAction]]:
    """Return the list of legal action classes for ``node``.

    Mirrors BAT's logic:
      * Lookup by node type.
      * For TRANSFORMATION node: if ``columns_match=True`` then only End is
        allowed (don't waste budget revising a successful pipeline).
      * Drop any action whose class already appears on the path-to-root
        (prevents loops like Schema -> Identify -> Schema).
    """
    raw = list(_NODE_TO_ACTION_CLASSES.get(node.node_type, []))

    if node.node_type == MCTSNodeType.TRANSFORMATION and node.columns_match:
        raw = [EndAction]

    seen_classes: set[type[BaseAction]] = set()
    for ancestor in node.path_to_root():
        if ancestor.parent_action is not None:
            seen_classes.add(type(ancestor.parent_action))
    # End / Revision may legitimately repeat (the path produces them only at
    # the end); but according to BAT, an action type already used on the path
    # is not eligible again. We exclude only the *intermediate* action
    # classes to mirror the original behaviour while keeping End reachable.
    intermediate_classes = {
        SchemaMatchAction,
        IdentifyColumnFunctionsAction,
        TransformationAction,
        TransformationRevisionAction,
    }
    legal = [
        cls for cls in raw
        if cls not in (seen_classes & intermediate_classes)
    ]
    return legal


__all__ = [
    "ActionContext",
    "BaseAction",
    "SchemaMatchAction",
    "IdentifyColumnFunctionsAction",
    "TransformationAction",
    "TransformationRevisionAction",
    "EndAction",
    "get_valid_action_space_for_node",
    "expected_target_columns",
    "_column_similarity",
    "_sandbox_columns",
]
