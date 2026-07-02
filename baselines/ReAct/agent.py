"""Main ReAct agent loop (full-pipeline-YAML protocol).

Each turn the LLM submits a complete pipeline YAML; the agent

  1. Parses the YAML into a Pipeline object via ``Pipeline.from_yaml``.
  2. Normalises every step by passing its op_name through
     :func:`baselines.DeepPrep.tree_agent.chain_to_steps`, so context-aware
     defaults (column lists, target_col, ...) are filled in even if the LLM
     omitted them.
  3. Checks structural legality via :func:`is_legal`.
  4. Resets the sandbox to the original data and re-executes the FULL
     pipeline (because each turn submits a full pipeline, not a delta).
  5. Trains the downstream model on a small sample and obtains
     ``(fitness, val_metrics, error)``.
  6. Renders an <observation> covering status, schema, the val-set metrics
     dict, the primary fitness, and the running best.

The highest-fitness pipeline across all turns is kept as the final output.
The agent does NOT call SAGA's repair() inside the loop (that would mask
the LLM's mistakes); the top-level :class:`ReAct` runner does one final
repair as a safety net.
"""
from __future__ import annotations

import copy
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from baselines.common.pipeline import (
    DataContext,
    Pipeline,
    PipelineStep,
    build_default_params,
    default_target_for,
)
from baselines.common.pipeline_constraints import is_legal
from baselines.DeepPrep.tree_agent import ChainParseError, chain_to_steps
from baselines.DeepPrep.sandbox import Sandbox
from baselines.DeepPrep.llm_client import LLMClient

from .operator_catalog import format_op_descriptions
from .prompts import (
    SYSTEM_REACT,
    render_observation,
    render_retry_feedback,
    render_user_initial,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------
_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)
_PIPELINE_RE = re.compile(r"<pipeline>(.*?)</pipeline>", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_FENCED_YAML_RE = re.compile(
    r"^\s*```(?:ya?ml)?\s*\n(.*?)\n```\s*$", re.DOTALL | re.IGNORECASE
)


@dataclass
class ParsedReActAction:
    is_terminate: bool = False
    pipeline_yaml: Optional[str] = None
    thought: Optional[str] = None
    action_text: Optional[str] = None


def _strip_yaml_fence(text: str) -> str:
    m = _FENCED_YAML_RE.match(text.strip())
    if m:
        return m.group(1).strip()
    return text.strip()


def parse_react_response(text: str) -> ParsedReActAction:
    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else None

    action_m = _ACTION_RE.search(text)
    action_text = action_m.group(1).strip() if action_m else None
    if action_text and action_text.strip().lower().startswith("terminate"):
        return ParsedReActAction(
            is_terminate=True, thought=thought, action_text=action_text
        )

    pipe_m = _PIPELINE_RE.search(text)
    if pipe_m:
        return ParsedReActAction(
            is_terminate=False,
            pipeline_yaml=_strip_yaml_fence(pipe_m.group(1)),
            thought=thought,
        )

    return ParsedReActAction(
        is_terminate=False,
        pipeline_yaml=None,
        thought=thought,
        action_text=action_text,
    )


# ---------------------------------------------------------------------------
# Step normalisation: fill context-dependent defaults the LLM may have skipped
# ---------------------------------------------------------------------------
def _normalize_steps(
    raw_steps: list[PipelineStep],
    ctx: DataContext,
    rng: random.Random,
) -> tuple[list[PipelineStep], Optional[str]]:
    """Re-build each step through ``chain_to_steps`` so missing/illegal
    context-dependent params are auto-filled. Returns ``(steps, error)``.

    The LLM-supplied ``params`` and ``target`` overlay the auto-filled
    defaults, so the LLM can override sensibly while still benefiting from
    the dataset-aware fallback.
    """
    if not raw_steps:
        return [], "Empty pipeline."

    out: list[PipelineStep] = []
    for s in raw_steps:
        # Build defaults from the operator catalog + dataset context.
        defaults = build_default_params(s.op, ctx, rng)
        if defaults is None:
            return out, (
                f"Operator {s.op!r} cannot be applied in the current "
                f"context (e.g. missing required column or wrong task type)."
            )
        params = dict(defaults)
        for k, v in (s.params or {}).items():
            params[k] = v
        target = s.target or default_target_for(s.op, ctx.task_type)
        out.append(PipelineStep(op=s.op, target=target, params=params))
    return out, None


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class TurnRecord:
    turn: int
    thought: Optional[str]
    pipeline_yaml: Optional[str]      # raw text the LLM produced
    parsed_ops: list[str]             # op names after parsing
    pipeline_steps: list[dict]        # serialisable step list
    status: str                       # success / parse_error / sandbox_error / eval_error / legality_error / terminate
    error: Optional[str]
    fitness: Optional[float]
    metrics: dict
    obs_text: Optional[str]
    is_terminate: bool


@dataclass
class ReActRunResult:
    best_pipeline: Pipeline
    best_fitness: Optional[float]
    best_metrics: dict
    best_turn: Optional[int]
    success: bool                     # at least one turn produced a fitness
    n_turns: int
    n_errors: int
    transcript: list[dict] = field(default_factory=list)
    trajectory: list[TurnRecord] = field(default_factory=list)
    accumulated_yaml: list[str] = field(default_factory=list)
    last_pipeline: Optional[Pipeline] = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class ReActAgent:
    """Full-pipeline-YAML ReAct agent.

    Parameters
    ----------
    llm
        DeepPrep's :class:`LLMClient` (api or local backend).
    sandbox
        DeepPrep's :class:`Sandbox`. Reset before every turn so the full
        pipeline is executed on the original data.
    ctx
        SAGA :class:`DataContext` (numeric/categorical/list/text columns,
        target_col, time_col, ...). Drives ``_normalize_steps``.
    downstream_evaluator
        Callable Pipeline -> ``(fitness, metrics, error)``. Used for
        every turn's downstream training feedback.
    """

    def __init__(
        self,
        llm: LLMClient,
        sandbox: Sandbox,
        ctx: DataContext,
        downstream_evaluator: Callable[
            [Pipeline], tuple[Optional[float], dict, Optional[str]]
        ],
        *,
        yaml_example: str,
        max_turns: int = 6,
        max_retry_per_turn: int = 2,
        max_err_cnt: int = 5,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.llm = llm
        self.sandbox = sandbox
        self.ctx = ctx
        self.downstream_evaluator = downstream_evaluator
        self.yaml_example = yaml_example

        self.max_turns = max(1, int(max_turns))
        self.max_retry_per_turn = max(0, int(max_retry_per_turn))
        self.max_err_cnt = max(1, int(max_err_cnt))
        self.verbose = bool(verbose)

        self._rng = random.Random(seed)
        self.transcript: list[dict] = []
        self.trajectory: list[TurnRecord] = []
        self.accumulated_yaml: list[str] = []
        self.n_errors = 0

    # ------------------------------------------------------------------
    def _log(self, *args, **kwargs) -> None:
        if self.verbose:
            print("[ReAct.Agent]", *args, **kwargs)

    # ------------------------------------------------------------------
    def _summary_text(self) -> str:
        ctx = self.ctx
        return (
            f"task_type={ctx.task_type} dataset={ctx.data_name}\n"
            f"target_col={ctx.target_col} time_col={ctx.time_col} id_col={ctx.id_col}\n"
            f"#numeric_cols={len(ctx.numeric_cols)} "
            f"#categorical_cols={len(ctx.categorical_cols)} "
            f"#list_cols={len(ctx.list_cols)} #text_cols={len(ctx.text_cols)}\n"
            f"has_user_df={ctx.has_user_df} has_item_df={ctx.has_item_df} "
            f"aux_dfs={ctx.aux_dfs}"
        )

    # ------------------------------------------------------------------
    def _ordering_hint(self) -> str:
        from baselines.common.pipeline_constraints import _TABULAR_ORDER, _REC_ORDER
        from baselines.common.operator_catalog import CATALOG
        order = _REC_ORDER if self.ctx.task_type == "rec" else _TABULAR_ORDER
        by_cat: dict = {}
        for name, spec in CATALOG.items():
            if spec.task_type not in (self.ctx.task_type, "both"):
                continue
            by_cat.setdefault(spec.category, []).append(name)
        lines = []
        for i, cat in enumerate(order):
            ops = sorted(by_cat.get(cat, []))[:5]
            ops_str = ", ".join(ops) if ops else "(no ops)"
            lines.append(f"  {i+1:>2}. {cat.name}: {ops_str}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _build_initial_messages(self, root_obs_text: str) -> list[dict]:
        op_descriptions = format_op_descriptions(self.ctx.task_type)
        user = render_user_initial(
            task_type=self.ctx.task_type,
            data_name=self.ctx.data_name,
            summary_text=self._summary_text(),
            op_descriptions=op_descriptions,
            obs_text=root_obs_text,
            yaml_example=self.yaml_example,
            ordering_hint=self._ordering_hint(),
            max_turns=self.max_turns,
        )
        return [
            {"role": "system", "content": SYSTEM_REACT},
            {"role": "user", "content": user},
        ]

    # ------------------------------------------------------------------
    def _record_turn(
        self,
        *,
        turn: int,
        thought: Optional[str],
        pipeline_yaml: Optional[str],
        steps: list[PipelineStep],
        status: str,
        error: Optional[str],
        fitness: Optional[float],
        metrics: dict,
        obs_text: Optional[str],
        is_terminate: bool,
    ) -> TurnRecord:
        rec = TurnRecord(
            turn=turn,
            thought=thought,
            pipeline_yaml=pipeline_yaml,
            parsed_ops=[s.op for s in steps],
            pipeline_steps=[s.to_dict() for s in steps],
            status=status,
            error=error,
            fitness=fitness,
            metrics=dict(metrics or {}),
            obs_text=obs_text,
            is_terminate=is_terminate,
        )
        self.trajectory.append(rec)
        return rec

    # ------------------------------------------------------------------
    def run(self) -> ReActRunResult:
        # 1. Reset sandbox -> initial obs (used in the very first user msg).
        try:
            reset = self.sandbox.reset()
        except Exception as e:  # pragma: no cover - sandbox normally robust
            self._log(f"Sandbox reset failed: {e}.")
            return ReActRunResult(
                best_pipeline=Pipeline(),
                best_fitness=None,
                best_metrics={},
                best_turn=None,
                success=False,
                n_turns=0,
                n_errors=1,
                transcript=self.transcript,
                trajectory=self.trajectory,
                accumulated_yaml=self.accumulated_yaml,
                last_pipeline=None,
            )
        if not reset.success or reset.obs is None:
            return ReActRunResult(
                best_pipeline=Pipeline(),
                best_fitness=None,
                best_metrics={},
                best_turn=None,
                success=False,
                n_turns=0,
                n_errors=1,
                transcript=self.transcript,
                trajectory=self.trajectory,
                accumulated_yaml=self.accumulated_yaml,
                last_pipeline=None,
            )

        messages = self._build_initial_messages(reset.obs.text)

        best_pipeline: Optional[Pipeline] = None
        best_fitness: Optional[float] = None
        best_metrics: dict = {}
        best_turn: Optional[int] = None
        last_pipeline: Optional[Pipeline] = None
        terminated_explicitly = False

        for turn in range(1, self.max_turns + 1):
            self._log(f"--- turn {turn}/{self.max_turns} ---")
            if self.n_errors >= self.max_err_cnt:
                self._log(f"max_err_cnt reached ({self.n_errors}); stopping.")
                break

            parsed: Optional[ParsedReActAction] = None
            steps: list[PipelineStep] = []
            yaml_text: Optional[str] = None
            normalize_err: Optional[str] = None

            # Same-turn retries for parse / format errors.
            got_executable = False
            for retry in range(self.max_retry_per_turn + 1):
                try:
                    response = self.llm.chat(messages)
                except Exception as e:
                    self.n_errors += 1
                    self._log(f"  attempt {retry + 1}: LLM call failed: {e}")
                    if self.n_errors >= self.max_err_cnt:
                        break
                    continue

                messages.append({"role": "assistant", "content": response})
                self.transcript.append(
                    {"role": "assistant", "content": response, "turn": turn,
                     "retry": retry}
                )
                parsed = parse_react_response(response)

                if parsed.is_terminate:
                    break

                if parsed.pipeline_yaml is None:
                    err = (
                        "Reply is missing both <pipeline>...</pipeline> and "
                        "<action>Terminate</action>. Provide exactly one."
                    )
                    self.n_errors += 1
                    if retry < self.max_retry_per_turn:
                        messages.append({
                            "role": "user",
                            "content": render_retry_feedback(
                                err, self.max_retry_per_turn - retry
                            ),
                        })
                        continue
                    self._record_turn(
                        turn=turn, thought=parsed.thought,
                        pipeline_yaml=None, steps=[],
                        status="parse_error", error=err,
                        fitness=None, metrics={},
                        obs_text=None, is_terminate=False,
                    )
                    break  # cannot recover this turn

                yaml_text = parsed.pipeline_yaml
                self.accumulated_yaml.append(yaml_text)

                # Parse YAML -> Pipeline
                try:
                    pipe = Pipeline.from_yaml(yaml_text)
                except Exception as e:
                    err = f"YAML parse error: {type(e).__name__}: {e}"
                    self.n_errors += 1
                    if retry < self.max_retry_per_turn:
                        messages.append({
                            "role": "user",
                            "content": render_retry_feedback(
                                err, self.max_retry_per_turn - retry
                            ),
                        })
                        continue
                    self._record_turn(
                        turn=turn, thought=parsed.thought,
                        pipeline_yaml=yaml_text, steps=[],
                        status="parse_error", error=err,
                        fitness=None, metrics={},
                        obs_text=None, is_terminate=False,
                    )
                    break

                if not pipe.steps:
                    err = "Parsed pipeline contains no steps."
                    self.n_errors += 1
                    if retry < self.max_retry_per_turn:
                        messages.append({
                            "role": "user",
                            "content": render_retry_feedback(
                                err, self.max_retry_per_turn - retry
                            ),
                        })
                        continue
                    self._record_turn(
                        turn=turn, thought=parsed.thought,
                        pipeline_yaml=yaml_text, steps=[],
                        status="parse_error", error=err,
                        fitness=None, metrics={},
                        obs_text=None, is_terminate=False,
                    )
                    break

                # Normalise step params using DataContext defaults.
                norm_steps, norm_err = _normalize_steps(
                    list(pipe.steps), self.ctx, self._rng
                )
                if norm_err is not None:
                    self.n_errors += 1
                    if retry < self.max_retry_per_turn:
                        messages.append({
                            "role": "user",
                            "content": render_retry_feedback(
                                norm_err, self.max_retry_per_turn - retry
                            ),
                        })
                        continue
                    self._record_turn(
                        turn=turn, thought=parsed.thought,
                        pipeline_yaml=yaml_text, steps=norm_steps,
                        status="parse_error", error=norm_err,
                        fitness=None, metrics={},
                        obs_text=None, is_terminate=False,
                    )
                    normalize_err = norm_err
                    break

                steps = norm_steps
                got_executable = True
                break

            if parsed is not None and parsed.is_terminate:
                terminated_explicitly = True
                self._record_turn(
                    turn=turn, thought=parsed.thought,
                    pipeline_yaml=None, steps=[],
                    status="terminate", error=None,
                    fitness=None, metrics={},
                    obs_text=None, is_terminate=True,
                )
                self._log("LLM emitted Terminate; ending loop.")
                break

            if not got_executable:
                # We already recorded the turn (parse / normalize error) and
                # gave the LLM at least one retry chance. Move on.
                if self.n_errors >= self.max_err_cnt:
                    break
                # Still feed a non-empty observation so the LLM has context
                # for the next turn (the retry feedback messages are already
                # in the transcript).
                continue

            # Legality check (no auto-repair: keep ReAct's self-correction
            # semantics intact).
            tentative = Pipeline(steps=copy.deepcopy(steps))
            if not is_legal(tentative, self.ctx.task_type):
                err = (
                    "Pipeline violates the canonical category-order / "
                    "mandatory-op constraints. Reorder operators or pick "
                    "different ones."
                )
                self.n_errors += 1
                obs = render_observation(
                    status="legality_error",
                    parsed_ops=[s.op for s in steps],
                    error=err,
                    fitness=None,
                    val_metrics=None,
                    best_fitness=best_fitness,
                    best_turn=best_turn,
                    best_metrics=best_metrics,
                    turn=turn,
                    max_turns=self.max_turns,
                )
                messages.append({"role": "user", "content": obs})
                self._record_turn(
                    turn=turn, thought=parsed.thought if parsed else None,
                    pipeline_yaml=yaml_text, steps=steps,
                    status="legality_error", error=err,
                    fitness=None, metrics={},
                    obs_text=obs, is_terminate=False,
                )
                continue

            last_pipeline = Pipeline(steps=copy.deepcopy(steps))

            # Reset sandbox + execute the FULL pipeline.
            try:
                self.sandbox.reset()
            except Exception as e:  # pragma: no cover
                err = f"Sandbox reset failed: {type(e).__name__}: {e}"
                self.n_errors += 1
                obs = render_observation(
                    status="sandbox_error",
                    parsed_ops=[s.op for s in steps],
                    error=err,
                    best_fitness=best_fitness,
                    best_turn=best_turn,
                    best_metrics=best_metrics,
                    turn=turn,
                    max_turns=self.max_turns,
                )
                messages.append({"role": "user", "content": obs})
                self._record_turn(
                    turn=turn, thought=parsed.thought if parsed else None,
                    pipeline_yaml=yaml_text, steps=steps,
                    status="sandbox_error", error=err,
                    fitness=None, metrics={},
                    obs_text=obs, is_terminate=False,
                )
                continue

            exec_res = self.sandbox.execute_chain(steps)
            if not exec_res.success:
                err = exec_res.error or "sandbox execution failed"
                self.n_errors += 1
                obs = render_observation(
                    status="sandbox_error",
                    parsed_ops=[s.op for s in steps],
                    error=err,
                    best_fitness=best_fitness,
                    best_turn=best_turn,
                    best_metrics=best_metrics,
                    turn=turn,
                    max_turns=self.max_turns,
                )
                messages.append({"role": "user", "content": obs})
                self._record_turn(
                    turn=turn, thought=parsed.thought if parsed else None,
                    pipeline_yaml=yaml_text, steps=steps,
                    status="sandbox_error", error=err,
                    fitness=None, metrics={},
                    obs_text=obs, is_terminate=False,
                )
                continue

            schema_text = (
                exec_res.obs.text if exec_res.obs is not None else ""
            )

            # Downstream evaluation -> (fitness, metrics, error)
            try:
                fitness, metrics, eval_err = self.downstream_evaluator(
                    Pipeline(steps=copy.deepcopy(steps))
                )
            except Exception as e:  # pragma: no cover - safety net
                fitness, metrics, eval_err = None, {}, f"{type(e).__name__}: {e}"

            if fitness is None:
                self.n_errors += 1
                obs = render_observation(
                    status="eval_error",
                    parsed_ops=[s.op for s in steps],
                    schema_text=schema_text,
                    error=eval_err or "downstream training failed",
                    val_metrics=metrics,
                    best_fitness=best_fitness,
                    best_turn=best_turn,
                    best_metrics=best_metrics,
                    turn=turn,
                    max_turns=self.max_turns,
                )
                messages.append({"role": "user", "content": obs})
                self._record_turn(
                    turn=turn, thought=parsed.thought if parsed else None,
                    pipeline_yaml=yaml_text, steps=steps,
                    status="eval_error", error=eval_err,
                    fitness=None, metrics=metrics,
                    obs_text=obs, is_terminate=False,
                )
                continue

            # Success branch: update best-of-N.
            if best_fitness is None or fitness > best_fitness:
                best_pipeline = Pipeline(steps=copy.deepcopy(steps))
                best_fitness = fitness
                best_metrics = dict(metrics or {})
                best_turn = turn

            obs = render_observation(
                status="success",
                parsed_ops=[s.op for s in steps],
                schema_text=schema_text,
                fitness=fitness,
                val_metrics=metrics,
                best_fitness=best_fitness,
                best_turn=best_turn,
                best_metrics=best_metrics,
                turn=turn,
                max_turns=self.max_turns,
            )
            messages.append({"role": "user", "content": obs})
            self._record_turn(
                turn=turn, thought=parsed.thought if parsed else None,
                pipeline_yaml=yaml_text, steps=steps,
                status="success", error=None,
                fitness=fitness, metrics=metrics,
                obs_text=obs, is_terminate=False,
            )
            self._log(
                f"  turn {turn}: ops={[s.op for s in steps]} "
                f"fitness={fitness:.4f} best={best_fitness:.4f}"
            )

        # Pick the best pipeline; if none succeeded, fall back to the last
        # parseable pipeline (the top-level runner will repair it).
        success = best_pipeline is not None
        if best_pipeline is None:
            if last_pipeline is not None:
                best_pipeline = last_pipeline
            else:
                best_pipeline = Pipeline()

        return ReActRunResult(
            best_pipeline=best_pipeline,
            best_fitness=best_fitness,
            best_metrics=best_metrics,
            best_turn=best_turn,
            success=success,
            n_turns=len(self.trajectory),
            n_errors=self.n_errors,
            transcript=self.transcript,
            trajectory=self.trajectory,
            accumulated_yaml=self.accumulated_yaml,
            last_pipeline=last_pipeline,
        )


__all__ = [
    "ReActAgent",
    "ReActRunResult",
    "TurnRecord",
    "ParsedReActAction",
    "parse_react_response",
]
