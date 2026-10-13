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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from baselines.DeepPrep.tree_agent import ChainParseError, chain_to_steps
from baselines.DeepPrep.operator_catalog import mandatory_ops_for_context
from baselines.common.data_ctx import DataContext
from baselines.common.pipeline import (
    Pipeline,
    PipelineStep,
    assign_dag_structure,
    check_pipeline,
)

from . import prompts
from .node import MCTSNode, MCTSNodeType
from .llm_client import BATSearchTimeLimit
from .operator_catalog import (
    format_op_descriptions,
    format_mandatory_examples,
    function_family_descriptions,
)

if TYPE_CHECKING:
    from .sandbox import Sandbox
    from baselines.DeepPrep.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_PIPE_RE = re.compile(r"<pipeline>(.*?)</pipeline>", re.DOTALL)  # legacy
_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def _extract(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text or "")
    return m.group(1).strip() if m else None


def _safe_json_loads(s: Optional[str]):
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_json_response(text: str):
    text = text or ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "<think>" in text:
        return None  # Truncated reasoning is not a final answer.
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    match = _JSON_BLOCK_RE.search(text or "")
    payload = match.group(1).strip() if match else (text or "").strip().strip("`")
    return _safe_json_loads(payload)


def _pipeline_from_response(text: str) -> Optional[str]:
    data = _extract_json_response(text)
    if isinstance(data, dict):
        value = data.get("pipeline", data.get("code"))
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            calls = []
            for item in value:
                if isinstance(item, str):
                    calls.append(item)
                    continue
                if not isinstance(item, dict):
                    return None
                name = item.get("operator", item.get("op", item.get("name")))
                params = item.get("params", item.get("kwargs", {}))
                if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_]\w*", name):
                    return None
                if not isinstance(params, dict) or any(
                    not isinstance(key, str) or not key.isidentifier() for key in params
                ):
                    return None
                args = ", ".join(f"{key}={arg!r}" for key, arg in params.items())
                calls.append(f"{name}({args})")
            return " --> ".join(calls)
    # Compatibility with trajectories produced by the earlier adapter.
    return _extract(_PIPE_RE, text)


def _repair_pipeline_response(text: str, action_ctx: "ActionContext") -> Optional[str]:
    chain = _pipeline_from_response(text)
    if chain:
        return chain
    repair_prompt = (
        "Convert the following response into valid JSON with exactly two keys: "
        "chain_of_thought_reasoning and pipeline. pipeline must be one DPPBench "
        "operator chain ending in Terminate. Return JSON only.\n\n" + (text or "")
    )
    repair_prompt += "\n" + _summary_text(action_ctx.ctx)
    repair_prompt += "\n" + format_op_descriptions(action_ctx.ctx.task_type)
    repaired = action_ctx.llm.chat([{"role": "user", "content": repair_prompt}], max_tokens=1024)
    return _pipeline_from_response(repaired)


def _summary_text(ctx: DataContext) -> str:
    expected = expected_target_columns(ctx)
    expected_display = (
        expected if len(expected) <= 40
        else f"{len(expected)} columns; first 30={expected[:30]}; full names are in Source Tables"
    )
    return (
        f"task_type={ctx.task_type} dataset={ctx.data_name}\n"
        f"expected_target_columns={expected_display}\n"
        f"mandatory_operators={mandatory_ops_for_context(ctx)}\n"
        f"valid_mandatory_operator_examples:\n{format_mandatory_examples(ctx)}\n"
        "Use keyword arguments only, never positional arguments.\n"
        f"target_col={ctx.target_col} time_col={ctx.time_col} "
        f"id_col={ctx.id_col}\n"
        f"#numeric_cols={len(ctx.numeric_cols)} "
        f"#categorical_cols={len(ctx.categorical_cols)} "
        f"#list_cols={len(ctx.list_cols)} #text_cols={len(ctx.text_cols)}\n"
        f"has_user_df={ctx.has_user_df} has_item_df={ctx.has_item_df} "
        f"aux_dfs={ctx.aux_dfs}"
        "\nFor wide tables, use LabelEncode(cols=None) to encode all object columns, "
        "and HandleMV(cols=None) for applicable missing columns; do not enumerate "
        "hundreds of columns. Keep reasoning to one sentence."
    )


def _observation_text(obs) -> str:
    """Keep wide-table samples within a practical LLM context window."""
    if len(obs.text) <= 12000:
        return obs.text
    lines = []
    for name, profile in obs.tables.items():
        columns = profile["columns"]
        sampled = columns[:20]
        lines.append(
            f"table {name}: shape={profile['shape']} null_ratio={profile['null_ratio']:.4f}"
        )
        lines.append(f"columns ({len(columns)}): {columns}")
        lines.append("sampled_dtypes: " + repr({
            col: profile["dtypes"][col] for col in sampled
        }))
        lines.append(f"sample_rows_first_20_columns: " + repr([
            {col: row.get(col) for col in sampled} for row in profile["head"][:2]
        ]))
    return "\n".join(lines)


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
    """Official BAT target-column recall reward."""
    if not expected:
        return 1.0
    a = set(actual)
    e = set(expected)
    return float(len(a & e)) / float(len(e))


def _pipeline_key(steps: list[PipelineStep]) -> str:
    """Distinguish candidates by parameters as well as operator names."""
    return json.dumps([step.to_dict() for step in steps], sort_keys=True, default=str)


def _previous_pipeline_prompt(chains: list[str]) -> str:
    if not chains:
        return ""
    return (
        "\nPreviously tried chains for this task:\n"
        + "\n".join(f"- {chain}" for chain in chains[-3:])
        + "\nIf the data supports a meaningful alternative, propose a different "
          "legal chain; do not add irrelevant operators solely for novelty.\n"
    )


def _score_steps_structurally(
    steps: list[PipelineStep],
    action_ctx: "ActionContext",
) -> tuple[list[PipelineStep], Optional[bool], Optional[float], Optional[str]]:
    """Execute the unmodified candidate to provide BAT EPO feedback."""
    if len(steps) > action_ctx.max_chain_len:
        return list(steps), False, 0.0, (
            f"pipeline length {len(steps)} exceeds {action_ctx.max_chain_len}"
        )
    pipe = Pipeline(steps=list(steps))
    assign_dag_structure(pipe, action_ctx.ctx, action_ctx.rng)
    report = check_pipeline(pipe, action_ctx.ctx)
    if not report.valid:
        return list(steps), False, 0.0, "; ".join(report.errors)
    initial = getattr(action_ctx.sandbox, "_initial_snapshot", None)
    if initial is not None:
        action_ctx.sandbox.restore(initial)
    exec_res = action_ctx.sandbox.execute_chain(pipe.steps)
    if not exec_res.success:
        if initial is not None:
            action_ctx.sandbox.restore(initial)
        return list(pipe.steps), False, 0.0, exec_res.error
    actual = _sandbox_columns(action_ctx.sandbox)
    expected = expected_target_columns(action_ctx.ctx)
    sim = _column_similarity(actual, expected)
    if initial is not None:
        action_ctx.sandbox.restore(initial)
    return list(steps), set(actual) == set(expected), sim, None


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
    n_action_samples: int = 1
    verbose: bool = False
    previous_pipeline_texts: list[str] = field(default_factory=list)
    downstream_evaluator: Any = None


def evaluate_downstream_feedback(node: MCTSNode, action_ctx: ActionContext) -> None:
    """Evaluate before expansion so revision sees validation-training failures."""
    if action_ctx.downstream_evaluator is None or node.exec_error:
        return
    steps = node.latest_pipeline_steps()
    if not steps:
        return
    pipe = Pipeline(steps=list(steps))
    assign_dag_structure(pipe, action_ctx.ctx, action_ctx.rng)
    if not check_pipeline(pipe, action_ctx.ctx).valid:
        return
    fitness, metrics, error = action_ctx.downstream_evaluator.evaluate_for_agent(pipe)
    node.downstream_fitness = fitness
    node.downstream_metrics = metrics
    node.downstream_error = error


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
            obs_text=_observation_text(obs),
            expected_target_columns=expected_target_columns(ctx),
        )
        nodes = []
        for _ in range(1):
            response = action_ctx.llm.chat(
                [
                    {"role": "system", "content": prompts.SYSTEM_BAT},
                    {"role": "user", "content": prompt},
                ], max_tokens=384,
            )
            nodes.append(MCTSNode(
                node_type=MCTSNodeType.SCHEMA_MATCH,
                parent=parent,
                parent_action=self,
                depth=parent.depth + 1,
                schema_match=_extract_json_response(response) or [],
            ))
        return nodes


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
            obs_text=_observation_text(obs),
            schema_match_text=schema_match_text,
            family_descriptions=function_family_descriptions(ctx.task_type),
        )
        responses = []
        for _ in range(1):
            response = action_ctx.llm.chat(
                [
                    {"role": "system", "content": prompts.SYSTEM_BAT},
                    {"role": "user", "content": prompt},
                ], max_tokens=384,
            )
            if response not in responses:
                responses.append(response)
        return [
            MCTSNode(
                node_type=MCTSNodeType.IDENTIFY_COLUMN_FUNCTIONS,
                parent=parent,
                parent_action=self,
                depth=parent.depth + 1,
                column_functions=response,
            )
            for response in responses
        ]


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
            obs_text=_observation_text(obs),
            op_descriptions=format_op_descriptions(ctx.task_type),
            schema_match_text=schema_match_text,
            column_functions_text=col_func_text,
            max_chain_len=action_ctx.max_chain_len,
        )
        prompt += _previous_pipeline_prompt(action_ctx.previous_pipeline_texts or [])
        nodes = []
        seen: set[str] = set()
        for _ in range(max(1, action_ctx.n_action_samples)):
            try:
                response = action_ctx.llm.chat(
                    [
                        {"role": "system", "content": prompts.SYSTEM_BAT},
                        {"role": "user", "content": prompt},
                    ], max_tokens=1024,
                )
            except BATSearchTimeLimit:
                if nodes:
                    break
                raise
            chain_str = _repair_pipeline_response(response, action_ctx) or ""
            steps, parse_error = _parse_chain_safe(chain_str, ctx, action_ctx.rng)
            if parse_error is None:
                steps, columns_match, column_similarity, exec_error = _score_steps_structurally(
                    steps, action_ctx
                )
            else:
                columns_match = False
                column_similarity = 0.0
                exec_error = parse_error
            key = _pipeline_key(steps) if steps else chain_str
            if key in seen and nodes:
                continue
            seen.add(key)
            child = MCTSNode(
                node_type=MCTSNodeType.TRANSFORMATION,
                parent=parent,
                parent_action=self,
                depth=parent.depth + 1,
                pipeline_steps=steps,
                pipeline_text=chain_str,
                columns_match=columns_match,
                column_similarity=column_similarity,
                exec_error=exec_error,
            )
            nodes.append(child)
            prompt += _previous_pipeline_prompt([chain_str])
        return nodes


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
        original_ops = [
            f"{s.op}(" + ", ".join(f"{k}={v!r}" for k, v in s.params.items()) + ")"
            for s in parent.latest_pipeline_steps()
        ]
        if not original_ops and parent.pipeline_text:
            original_ops = [parent.pipeline_text]
        prompt = prompts.render_transformation_revision(
            task_type=ctx.task_type,
            data_name=ctx.data_name,
            op_descriptions=format_op_descriptions(ctx.task_type),
            summary_text=_summary_text(ctx),
            obs_after_exec=_observation_text(obs),
            original_pipeline_ops=original_ops,
            columns_match=parent.columns_match,
            column_similarity=parent.column_similarity,
            downstream_fitness=parent.downstream_fitness,
            downstream_metrics=parent.downstream_metrics,
            exec_error=parent.exec_error,
            downstream_error=parent.downstream_error,
            max_chain_len=action_ctx.max_chain_len,
        )
        prompt += _previous_pipeline_prompt(action_ctx.previous_pipeline_texts or [])
        nodes = []
        seen: set[str] = set()
        for _ in range(max(1, action_ctx.n_action_samples)):
            try:
                response = action_ctx.llm.chat(
                    [
                        {"role": "system", "content": prompts.SYSTEM_BAT},
                        {"role": "user", "content": prompt},
                    ], max_tokens=1024,
                )
            except BATSearchTimeLimit:
                if nodes:
                    break
                raise
            chain_str = _repair_pipeline_response(response, action_ctx) or ""
            steps, parse_error = _parse_chain_safe(chain_str, ctx, action_ctx.rng)
            if parse_error is None:
                steps, columns_match, column_similarity, exec_error = _score_steps_structurally(
                    steps, action_ctx
                )
            else:
                columns_match = False
                column_similarity = 0.0
                exec_error = parse_error
            key = _pipeline_key(steps) if steps else chain_str
            if key in seen and nodes:
                continue
            seen.add(key)
            child = MCTSNode(
                node_type=MCTSNodeType.REVISED_TRANSFORMATION,
                parent=parent,
                parent_action=self,
                depth=parent.depth + 1,
                revised_pipeline_steps=steps,
                pipeline_text=chain_str,
                columns_match=columns_match,
                column_similarity=column_similarity,
                exec_error=exec_error,
            )
            nodes.append(child)
            prompt += _previous_pipeline_prompt([chain_str])
        return nodes


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
class EndAction(BaseAction):
    name = "End"
    target_node_type = MCTSNodeType.END

    def create_children_nodes(self, parent, action_ctx):
        # Preserve the exact LLM candidate. DPPBench legality is a hard gate;
        # unlike finalize_pipeline(), this never inserts or rewrites operators.
        steps = parent.latest_pipeline_steps()
        pipe = Pipeline(steps=list(steps))
        assign_dag_structure(pipe, action_ctx.ctx, action_ctx.rng)
        report = check_pipeline(pipe, action_ctx.ctx)
        child = MCTSNode(
            node_type=MCTSNodeType.END,
            parent=parent,
            parent_action=self,
            depth=parent.depth + 1,
            final_pipeline_steps=list(steps),
            exec_error=None if report.valid else "; ".join(report.errors),
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
        steps = chain_to_steps(chain_str, ctx, rng, strict_instances=True)
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

    if node.node_type == MCTSNodeType.TRANSFORMATION and node.columns_match and not node.downstream_error:
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
