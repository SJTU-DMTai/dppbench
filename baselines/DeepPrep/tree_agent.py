"""Tree-based Agentic Reasoning loop for DeepPrep.

Implements the high-level multi-turn protocol described in the original
paper: the LLM iteratively proposes short operator chains, observes the
sandbox feedback during ``<operator>`` exploration, and finally emits a
``<solution>`` chain. When downstream feedback is enabled, successful
``<solution>`` attempts also receive a small-data downstream metric. Failed
branches can be reverted via ``<backtrack/>``.

Operator parsing is intentionally permissive: the LLM only needs to write
``OpName`` (optionally followed by ``(arg=value, ...)``) and the agent fills
in the dataset-aware default parameters via ``baselines.SAGA.pipeline.make_step``.
"""
from __future__ import annotations

import ast
import copy
import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from baselines.common.operator_catalog import CATALOG, operators_for_task
from baselines.SAGA.pipeline import (
    DataContext,
    Pipeline,
    PipelineStep,
    build_default_params,
    default_target_for,
    make_step,
)
from baselines.SAGA.pipeline_constraints import is_legal, repair

from .llm_client import LLMClient
from .operator_catalog import format_op_descriptions
from .prompts import (
    SYSTEM_TREE_SEARCH,
    render_backtrack_hint,
    render_observation,
    render_solution_feedback,
    render_user_initial,
)
from .sandbox import Sandbox
from .tree_node import SearchTree


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------
_TAG_RE = {
    "think": re.compile(r"<think>(.*?)</think>", re.DOTALL),
    "plan": re.compile(r"<plan>(.*?)</plan>", re.DOTALL),
    "operator": re.compile(r"<operator>(.*?)</operator>", re.DOTALL),
    "solution": re.compile(r"<solution>(.*?)</solution>", re.DOTALL),
}
_BACKTRACK_RE = re.compile(r"<backtrack\s*/?>")


def _extract_tag(text: str, tag: str) -> Optional[str]:
    m = _TAG_RE[tag].search(text)
    if not m:
        return None
    return m.group(1).strip()


@dataclass
class ParsedAction:
    kind: str                              # "operator" | "solution" | "backtrack" | "noop"
    chain_str: Optional[str] = None
    think: Optional[str] = None
    plan: Optional[str] = None


def parse_response(text: str) -> ParsedAction:
    think = _extract_tag(text, "think")
    plan = _extract_tag(text, "plan")
    op = _extract_tag(text, "operator")
    sol = _extract_tag(text, "solution")
    if sol is not None:
        return ParsedAction("solution", chain_str=sol, think=think, plan=plan)
    if op is not None:
        return ParsedAction("operator", chain_str=op, think=think, plan=plan)
    if _BACKTRACK_RE.search(text):
        return ParsedAction("backtrack", think=think, plan=plan)
    return ParsedAction("noop", think=think, plan=plan)


# ---------------------------------------------------------------------------
# Operator-chain parsing: "Op1(arg=value, x=[1,2]) --> Op2 --> Terminate"
# ---------------------------------------------------------------------------
_OP_CALL_RE = re.compile(r"^([A-Za-z_][A-Za-z_0-9]*)\s*(\((.*)\))?$", re.DOTALL)


def _parse_arg_value(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return None
    # Try Python literal first (handles ints, floats, lists, dicts, tuples, bools, None, strings)
    try:
        return ast.literal_eval(raw)
    except Exception:
        pass
    # Try JSON
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Strip surrounding quotes, otherwise return raw string
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        return raw[1:-1]
    return raw


def _split_args(s: str) -> list[str]:
    """Split a comma-separated argument string while respecting nested
    brackets and quoted strings.
    """
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: Optional[str] = None
    for ch in s:
        if quote is not None:
            buf.append(ch)
            if ch == quote and (len(buf) < 2 or buf[-2] != "\\"):
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
            buf.append(ch)
            continue
        if ch in "([{":
            depth += 1
            buf.append(ch)
            continue
        if ch in ")]}":
            depth -= 1
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _parse_op_call(call_str: str) -> tuple[str, dict]:
    """Parse 'OpName(a=1, b=[1,2])' -> ('OpName', {'a': 1, 'b': [1,2]})."""
    s = call_str.strip()
    m = _OP_CALL_RE.match(s)
    if not m:
        raise ValueError(f"Could not parse operator call: {s!r}")
    op_name = m.group(1)
    inner = m.group(3) or ""
    params: dict = {}
    if inner.strip():
        for kv in _split_args(inner):
            if "=" in kv:
                k, v = kv.split("=", 1)
                params[k.strip()] = _parse_arg_value(v)
            else:
                params.setdefault("__positional__", []).append(_parse_arg_value(kv))
    return op_name, params


# ---------------------------------------------------------------------------
# Chain → list[PipelineStep]
# ---------------------------------------------------------------------------
@dataclass
class ChainParseError(Exception):
    message: str

    def __str__(self) -> str:  # pragma: no cover -- trivial
        return self.message


def chain_to_steps(
    chain_str: str,
    ctx: DataContext,
    rng: random.Random,
    *,
    drop_terminate: bool = True,
) -> list[PipelineStep]:
    """Convert ``Op1(...) --> Op2(...) --> Terminate`` into PipelineStep list.

    Unknown operator names / operators that do not apply to the task type
    raise ChainParseError. Missing context-dependent parameters are
    auto-filled via ``build_default_params``.
    """
    raw_calls = [c.strip() for c in chain_str.split("-->") if c.strip()]
    valid_ops = set(operators_for_task(ctx.task_type))
    steps: list[PipelineStep] = []
    for raw in raw_calls:
        op_name, llm_params = _parse_op_call(raw)
        if op_name == "Terminate":
            if drop_terminate:
                continue
            steps.append(PipelineStep(op="Terminate", target="both", params={}))
            continue
        if op_name not in CATALOG:
            raise ChainParseError(
                f"Unknown operator {op_name!r}. "
                f"Use only operators from the whitelist."
            )
        if op_name not in valid_ops:
            raise ChainParseError(
                f"Operator {op_name!r} is not applicable to task_type "
                f"{ctx.task_type!r}."
            )
        # Build default params, then overlay LLM-supplied params (so the
        # LLM can override sensibly while still benefiting from the
        # context-aware defaults for column lists etc.).
        defaults = build_default_params(op_name, ctx, rng)
        if defaults is None:
            raise ChainParseError(
                f"Operator {op_name!r} cannot be applied in the current "
                f"context (e.g. missing required column)."
            )
        params = dict(defaults)
        for k, v in llm_params.items():
            if k == "__positional__":
                continue
            params[k] = v
        target = default_target_for(op_name, ctx.task_type)
        steps.append(PipelineStep(op=op_name, target=target, params=params))
    return steps


# ---------------------------------------------------------------------------
# Tree agent
# ---------------------------------------------------------------------------
@dataclass
class AgentRunResult:
    pipeline: Pipeline
    success: bool
    n_turns: int
    n_errors: int
    transcript: list[dict] = field(default_factory=list)
    tree: dict = field(default_factory=dict)
    solution_attempts: list[dict] = field(default_factory=list)


class TreeAgent:
    def __init__(
        self,
        llm: LLMClient,
        sandbox: Sandbox,
        ctx: DataContext,
        max_explore_turn: int = 5,
        max_chain_len: int = 6,
        max_depth: int = 8,
        max_err_cnt: int = 5,
        verbose: bool = True,
        seed: int = 42,
        downstream_evaluator: Optional[
            Callable[[Pipeline], tuple[Optional[float], dict, Optional[str]]]
        ] = None,
        max_solution_attempts: int = 3,
    ) -> None:
        self.llm = llm
        self.sandbox = sandbox
        self.ctx = ctx
        self.max_explore_turn = int(max_explore_turn)
        self.max_chain_len = int(max_chain_len)
        self.max_depth = int(max_depth)
        self.max_err_cnt = int(max_err_cnt)
        self.verbose = bool(verbose)
        self._rng = random.Random(seed)
        self.tree = SearchTree()
        self.transcript: list[dict] = []
        self.n_errors = 0
        self.downstream_evaluator = downstream_evaluator
        self.max_solution_attempts = int(max_solution_attempts)
        self.solution_attempts: list[dict] = []

    # ------------------------------------------------------------------
    def _log(self, *args, **kwargs) -> None:
        if self.verbose:
            print("[DeepPrep.Agent]", *args, **kwargs)

    # ------------------------------------------------------------------
    def _best_attempt_fitness(self) -> Optional[float]:
        fits = [
            a["fitness"] for a in self.solution_attempts
            if isinstance(a.get("fitness"), float)
        ]
        return max(fits) if fits else None

    def _select_best_attempt_pipeline(self) -> Optional[Pipeline]:
        best: Optional[dict] = None
        for a in self.solution_attempts:
            if not isinstance(a.get("fitness"), float):
                continue
            if best is None or a["fitness"] > best["fitness"]:
                best = a
        if best is None:
            return None
        return best["pipeline"]

    # ------------------------------------------------------------------
    def _build_initial_messages(self, root_obs: str) -> list[dict]:
        op_descriptions = format_op_descriptions(self.ctx.task_type)
        summary = self._summary_text()
        user = render_user_initial(
            task_type=self.ctx.task_type,
            data_name=self.ctx.data_name,
            summary_text=summary,
            op_descriptions=op_descriptions,
            obs_text=root_obs,
            max_chain_len=self.max_chain_len,
            max_explore_turn=self.max_explore_turn,
        )
        return [
            {"role": "system", "content": SYSTEM_TREE_SEARCH},
            {"role": "user", "content": user},
        ]

    def _summary_text(self) -> str:
        ctx = self.ctx
        return (
            f"task_type={ctx.task_type} dataset={ctx.data_name}\n"
            f"target_col={ctx.target_col} time_col={ctx.time_col} id_col={ctx.id_col}\n"
            f"#numeric_cols={len(ctx.numeric_cols)} #categorical_cols={len(ctx.categorical_cols)} "
            f"#list_cols={len(ctx.list_cols)} #text_cols={len(ctx.text_cols)}\n"
            f"has_user_df={ctx.has_user_df} has_item_df={ctx.has_item_df} aux_dfs={ctx.aux_dfs}"
        )

    # ------------------------------------------------------------------
    def run(self) -> AgentRunResult:
        # 1. Reset sandbox -> root obs
        reset = self.sandbox.reset()
        if not reset.success or reset.obs is None:
            # Extremely unlikely; fall back to a repair-only pipeline.
            self._log("Sandbox reset failed; returning repair-only pipeline.")
            pipe = Pipeline()
            repair(pipe, self.ctx.task_type, self.ctx)
            return AgentRunResult(
                pipeline=pipe, success=False, n_turns=0,
                n_errors=1, transcript=self.transcript,
                tree=self.tree.to_dict(),
            )

        self.tree.add_root(snapshot=reset.snapshot, obs_text=reset.obs.text)

        messages = self._build_initial_messages(reset.obs.text)

        final_pipeline: Optional[Pipeline] = None
        turn = 0
        for turn_idx in range(1, self.max_explore_turn + 2):
            turn = turn_idx
            self._log(f"--- turn {turn_idx}/{self.max_explore_turn + 1} ---")
            try:
                response = self.llm.chat(messages)
            except Exception as e:
                self._log(f"LLM call failed: {e}")
                self.n_errors += 1
                if self.n_errors >= self.max_err_cnt:
                    break
                # On retry, drop last user msg if any
                continue

            messages.append({"role": "assistant", "content": response})
            self.transcript.append({"role": "assistant", "content": response, "turn": turn_idx})
            action = parse_response(response)
            self._log(f"action={action.kind}")

            if action.kind == "noop":
                self.n_errors += 1
                messages.append({
                    "role": "user",
                    "content": (
                        "<observation>FORMAT ERROR: your reply did not contain "
                        "<operator>, <solution>, or <backtrack/>. Please retry "
                        "and end with exactly one action tag.</observation>"
                    ),
                })
                if self.n_errors >= self.max_err_cnt:
                    break
                continue

            if action.kind == "backtrack":
                back = self.tree.backtrack()
                if back is None:
                    messages.append({
                        "role": "user",
                        "content": (
                            "<observation>Already at the root; cannot backtrack. "
                            "Use <operator> or <solution> instead.</observation>"
                        ),
                    })
                else:
                    parent_obs = back.obs_text
                    removed_ops: list[str] = []
                    # Restore sandbox state
                    self.sandbox.restore(back.snapshot)  # type: ignore[arg-type]
                    msg = render_backtrack_hint(parent_obs, removed_ops)
                    messages.append({"role": "user", "content": msg})
                continue

            # ----- operator / solution: parse the chain -----
            try:
                steps = chain_to_steps(action.chain_str or "", self.ctx, self._rng)
            except ChainParseError as e:
                self.n_errors += 1
                messages.append({
                    "role": "user",
                    "content": render_observation(
                        obs_text="",
                        error=f"Chain parse error: {e.message}",
                        turn_left=self.max_explore_turn - turn_idx + 1,
                        max_explore_turn=self.max_explore_turn,
                    ),
                })
                if self.n_errors >= self.max_err_cnt:
                    break
                continue

            if action.kind == "operator":
                if len(steps) > self.max_chain_len:
                    self.n_errors += 1
                    messages.append({
                        "role": "user",
                        "content": render_observation(
                            obs_text="",
                            error=(
                                f"Chain length ({len(steps)}) exceeds "
                                f"max_chain_len={self.max_chain_len}."
                            ),
                            turn_left=self.max_explore_turn - turn_idx + 1,
                            max_explore_turn=self.max_explore_turn,
                        ),
                    })
                    continue
                # Tentative legality check on cumulative pipeline
                cur_node = self.tree.cursor
                tentative = Pipeline(steps=cur_node.accumulated_steps + steps)
                if not is_legal(tentative, self.ctx.task_type):
                    self.n_errors += 1
                    messages.append({
                        "role": "user",
                        "content": render_observation(
                            obs_text="",
                            error=(
                                "Pipeline order violates DeepPrep constraints "
                                "(category ordering / required ops / duplicates). "
                                "Reorder operators or backtrack."
                            ),
                            turn_left=self.max_explore_turn - turn_idx + 1,
                            max_explore_turn=self.max_explore_turn,
                        ),
                    })
                    continue
                # Execute
                exec_res = self.sandbox.execute_chain(steps)
                if not exec_res.success:
                    self.n_errors += 1
                    messages.append({
                        "role": "user",
                        "content": render_observation(
                            obs_text="",
                            error=exec_res.error,
                            turn_left=self.max_explore_turn - turn_idx + 1,
                            max_explore_turn=self.max_explore_turn,
                        ),
                    })
                    if self.n_errors >= self.max_err_cnt:
                        break
                    continue
                # Add child node and advance cursor
                self.tree.add_child(
                    new_steps=steps,
                    snapshot=exec_res.snapshot,           # type: ignore[arg-type]
                    obs_text=exec_res.obs.text,           # type: ignore[union-attr]
                )
                if self.tree.cursor.depth >= self.max_depth:
                    # Force a solution emission at the next turn.
                    messages.append({
                        "role": "user",
                        "content": render_observation(
                            obs_text=exec_res.obs.text,   # type: ignore[union-attr]
                            error=None,
                            turn_left=1,
                            max_explore_turn=self.max_explore_turn,
                        ),
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": render_observation(
                            obs_text=exec_res.obs.text,   # type: ignore[union-attr]
                            error=None,
                            turn_left=self.max_explore_turn - turn_idx + 1,
                            max_explore_turn=self.max_explore_turn,
                        ),
                    })
                continue

            if action.kind == "solution":
                # The solution chain is the FULL pipeline — start from a fresh
                # sandbox snapshot (root) so we don't double-apply earlier ops.
                root_snap = self.tree.nodes[self.tree.root_id].snapshot  # type: ignore[index]
                self.sandbox.restore(root_snap)                          # type: ignore[arg-type]
                pipe = Pipeline(steps=steps)
                # Repair to satisfy mandatory constraints (rec mandatory ops,
                # tabular tail). This guarantees an executable pipeline even
                # if the LLM forgot a required operator.
                repair(pipe, self.ctx.task_type, self.ctx)
                exec_res = self.sandbox.execute_chain(pipe.steps)
                if not exec_res.success:
                    self.n_errors += 1
                    messages.append({
                        "role": "user",
                        "content": render_observation(
                            obs_text="",
                            error=(
                                "Final solution failed to execute end-to-end: "
                                f"{exec_res.error}"
                            ),
                            turn_left=self.max_explore_turn - turn_idx + 1,
                            max_explore_turn=self.max_explore_turn,
                        ),
                    })
                    if self.n_errors >= self.max_err_cnt:
                        break
                    continue

                # Structurally-valid solution. If no downstream evaluator is
                # configured, accept it immediately (legacy behaviour).
                if self.downstream_evaluator is None:
                    final_pipeline = pipe
                    self._log(f"solution accepted ({len(pipe)} steps).")
                    break

                # Otherwise: run the downstream model on a small subsample
                # and feed the metric back to the LLM.
                fitness, metrics, eval_err = self.downstream_evaluator(pipe)
                attempt_idx = len(self.solution_attempts) + 1
                attempt_record = {
                    "attempt": attempt_idx,
                    "ops": [s.op for s in pipe.steps],
                    "fitness": fitness,
                    "metrics": metrics,
                    "error": eval_err,
                    "pipeline": copy.deepcopy(pipe),
                }
                self.solution_attempts.append(attempt_record)
                best_fit = self._best_attempt_fitness()
                self._log(
                    f"solution attempt {attempt_idx}/{self.max_solution_attempts} "
                    f"fitness={fitness} err={eval_err}"
                )
                # Decide whether to stop attempting solutions.
                attempts_done = attempt_idx >= self.max_solution_attempts
                turn_left = self.max_explore_turn - turn_idx + 1
                if attempts_done or turn_left <= 0:
                    final_pipeline = self._select_best_attempt_pipeline()
                    if final_pipeline is None:
                        # All attempts had downstream errors; keep this
                        # structurally-valid pipeline as the fallback.
                        final_pipeline = pipe
                    self._log(
                        f"solution loop done (attempts={attempt_idx}, "
                        f"best_fitness={best_fit})."
                    )
                    break
                messages.append({
                    "role": "user",
                    "content": render_solution_feedback(
                        pipeline_ops=[s.op for s in pipe.steps],
                        fitness=fitness,
                        metrics=metrics,
                        error=eval_err,
                        attempt_idx=attempt_idx,
                        max_attempts=self.max_solution_attempts,
                        best_fitness=best_fit,
                    ),
                })
                continue

        # Fallback: if no solution was produced, build a minimal repair-only
        # pipeline so the rest of the system can still output something.
        success = final_pipeline is not None
        if final_pipeline is None:
            self._log("agent exhausted turns without a solution; using repair fallback.")
            final_pipeline = Pipeline()
            repair(final_pipeline, self.ctx.task_type, self.ctx)

        # Build a serialisable view of the solution attempts (drop pipeline
        # objects so the dict is JSON-friendly).
        attempts_serialisable = [
            {
                "attempt": a["attempt"],
                "ops": a["ops"],
                "fitness": a["fitness"],
                "metrics": a["metrics"],
                "error": a["error"],
            }
            for a in self.solution_attempts
        ]

        return AgentRunResult(
            pipeline=final_pipeline,
            success=success,
            n_turns=turn,
            n_errors=self.n_errors,
            transcript=self.transcript,
            tree=self.tree.to_dict(),
            solution_attempts=attempts_serialisable,
        )


__all__ = [
    "TreeAgent",
    "AgentRunResult",
    "ParsedAction",
    "parse_response",
    "chain_to_steps",
    "ChainParseError",
]
