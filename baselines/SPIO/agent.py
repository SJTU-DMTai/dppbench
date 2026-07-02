"""SPIO agent: NL plan + per-stage best-of-N CustomOp selection.

The agent operates in two phases:

  1. ``generate_plan()``  — a single LLM call that drafts a strategy for
     each of the 5 stages (integration / cleaning / preprocessing /
     feature_engineering / transformation). Only the 4 single-table
     stages will be turned into code; the integration plan is recorded
     for completeness.
  2. ``run()``            — for each code stage in order, observe the
     current data state via the sandbox, ask the LLM for ``n_candidates``
     ``CustomOp`` snippets, evaluate each by composing it onto the
     already-chosen prefix and running downstream training, then commit
     the highest-AUC candidate as the prefix for the next stage. This
     greedy single-path selection is SPIO-S.

Failure handling: snippets that fail to parse or whose downstream
evaluation returns ``None`` are recorded with ``fitness=None``. If every
candidate fails for a stage, that stage is skipped (no CustomOp added).
"""
from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from baselines.DeepPrep.llm_client import LLMClient
from baselines.DeepPrep.sandbox import Sandbox
from baselines.common.pipeline import DataContext, Pipeline, PipelineStep

from .prompts import (
    SYSTEM_SPIO,
    parse_code_blocks,
    parse_plan_sections,
    render_codegen_prompt,
    render_plan_prompt,
)
from .stages import CODE_STAGES, STAGES, stage_op_descriptions


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class CandidateRecord:
    code: Optional[str]
    fitness: Optional[float]
    metrics: dict
    error: Optional[str]


@dataclass
class StageRecord:
    stage: str
    plan: Optional[str]
    obs_text: Optional[str]
    candidates: list[CandidateRecord] = field(default_factory=list)
    chosen_index: Optional[int] = None
    chosen_fitness: Optional[float] = None
    chosen_code: Optional[str] = None


@dataclass
class SPIORunResult:
    nl_plan: dict[str, str]
    chosen_prefix: list[PipelineStep]
    stage_records: list[StageRecord]
    best_fitness_in_loop: Optional[float]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
ScaffoldFn = Callable[[list[PipelineStep]], Pipeline]
EvalFn = Callable[[Pipeline], tuple[Optional[float], dict, Optional[str]]]


class SPIOAgent:
    """Per-stage CustomOp selector following the SPIO-S protocol."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        sandbox: Sandbox,
        ctx: DataContext,
        evaluate_fn: EvalFn,
        build_scaffold_fn: ScaffoldFn,
        build_prefix_scaffold_fn: Optional[ScaffoldFn] = None,
        n_candidates: int = 2,
        max_retry: int = 2,
        stage_max_per_cat: int = 6,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.llm = llm
        self.sandbox = sandbox
        self.ctx = ctx
        self.evaluate_fn = evaluate_fn
        self.build_scaffold_fn = build_scaffold_fn
        self.build_prefix_scaffold_fn = build_prefix_scaffold_fn or build_scaffold_fn
        self.n_candidates = max(1, int(n_candidates))
        self.max_retry = max(0, int(max_retry))
        self.stage_max_per_cat = max(1, int(stage_max_per_cat))
        self._rng = random.Random(seed)
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------
    def _log(self, *args, **kwargs) -> None:
        if self.verbose:
            print("[SPIO.Agent]", *args, **kwargs)

    # ------------------------------------------------------------------
    def _summary_text(self) -> str:
        ctx = self.ctx
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

    def _stage_target(self) -> str:
        return "interaction" if self.ctx.task_type == "rec" else "both"

    # ------------------------------------------------------------------
    def _observe_prefix(self, prefix_steps: list[PipelineStep]) -> str:
        """Replay the scaffolded prefix in the sandbox and return obs text."""
        self.sandbox.reset()
        scaffolded = self.build_prefix_scaffold_fn(
            [copy.deepcopy(s) for s in prefix_steps]
        )
        if not scaffolded.steps:
            obs = self.sandbox._observe()
            return obs.text
        try:
            res = self.sandbox.execute_chain(scaffolded.steps)
        except Exception as e:  # pragma: no cover - safety net
            return f"(sandbox error while replaying prefix: {type(e).__name__}: {e})"
        if not res.success or res.obs is None:
            return f"(sandbox error: {res.error or 'unknown'})"
        return res.obs.text

    # ------------------------------------------------------------------
    def generate_plan(self, init_obs_text: str) -> dict[str, str]:
        stage_hints = {
            stage: stage_op_descriptions(
                stage, self.ctx.task_type, max_per_cat=self.stage_max_per_cat
            )
            for stage in STAGES
        }
        prompt = render_plan_prompt(
            task_type=self.ctx.task_type,
            data_name=self.ctx.data_name,
            summary_text=self._summary_text(),
            obs_text=init_obs_text,
            stage_hints=stage_hints,
        )
        messages = [
            {"role": "system", "content": SYSTEM_SPIO},
            {"role": "user", "content": prompt},
        ]

        plan: dict[str, str] = {}
        last_err: Optional[str] = None
        for attempt in range(self.max_retry + 1):
            try:
                reply = self.llm.chat(messages)
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                self._log(f"[plan] LLM call failed (attempt {attempt + 1}): {last_err}")
                continue
            sections = parse_plan_sections(reply)
            if sections:
                plan = sections
                break
            last_err = "no '## Stage N: <stage>' headers found in plan reply"
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": (
                    "Your previous reply didn't contain the required "
                    "'## Stage N: <stage>' headers. Please re-emit the "
                    "plan exactly in the requested 5-section format."
                ),
            })

        if not plan:
            self._log(f"[plan] all attempts failed; using empty plan ({last_err})")
        # Fill in any missing stages with placeholder text so the codegen
        # prompt always has something to anchor on.
        for stage in STAGES:
            plan.setdefault(stage, "(no plan provided)")
        return plan

    # ------------------------------------------------------------------
    def _wrap_candidate(self, code: str) -> PipelineStep:
        return PipelineStep(
            op="CustomOp",
            target=self._stage_target(),
            params={"code": code, "entry": "pipeline"},
        )

    # ------------------------------------------------------------------
    def _generate_candidates(
        self,
        *,
        stage: str,
        stage_plan: str,
        obs_text: str,
    ) -> list[str]:
        prompt = render_codegen_prompt(
            task_type=self.ctx.task_type,
            data_name=self.ctx.data_name,
            stage=stage,
            stage_plan=stage_plan,
            summary_text=self._summary_text(),
            obs_text=obs_text,
            n_candidates=self.n_candidates,
        )
        messages = [
            {"role": "system", "content": SYSTEM_SPIO},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(self.max_retry + 1):
            try:
                reply = self.llm.chat(messages)
            except Exception as e:
                self._log(
                    f"[{stage}] LLM call failed (attempt {attempt + 1}): "
                    f"{type(e).__name__}: {e}"
                )
                continue
            blocks = parse_code_blocks(reply)
            if blocks:
                return blocks[: self.n_candidates]
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": (
                    "Your previous reply contained no fenced ```python ...``` "
                    "code blocks. Reply again with EXACTLY "
                    f"{self.n_candidates} fenced ```python``` blocks, each "
                    "defining ``def pipeline(df): ... return df``."
                ),
            })
        return []

    # ------------------------------------------------------------------
    def _select_stage(
        self,
        *,
        stage: str,
        stage_plan: str,
        chosen_prefix: list[PipelineStep],
    ) -> StageRecord:
        obs_text = self._observe_prefix(chosen_prefix)
        rec = StageRecord(stage=stage, plan=stage_plan, obs_text=obs_text)

        codes = self._generate_candidates(
            stage=stage, stage_plan=stage_plan, obs_text=obs_text
        )
        if not codes:
            self._log(f"[{stage}] no parsable candidates; skipping stage.")
            return rec

        for i, code in enumerate(codes):
            cand_step = self._wrap_candidate(code)
            try:
                pipeline = self.build_scaffold_fn(
                    [copy.deepcopy(s) for s in chosen_prefix] + [cand_step]
                )
            except Exception as e:  # pragma: no cover - scaffold is robust
                rec.candidates.append(CandidateRecord(
                    code=code, fitness=None, metrics={},
                    error=f"scaffold error: {type(e).__name__}: {e}",
                ))
                continue

            try:
                fitness, metrics, err = self.evaluate_fn(pipeline)
            except Exception as e:  # pragma: no cover - safety net
                fitness, metrics, err = None, {}, f"{type(e).__name__}: {e}"

            rec.candidates.append(CandidateRecord(
                code=code,
                fitness=float(fitness) if isinstance(fitness, (int, float)) else None,
                metrics=dict(metrics or {}),
                error=err,
            ))
            fit_str = (
                f"{fitness:.4f}"
                if isinstance(fitness, (int, float)) and fitness is not None
                else "FAIL"
            )
            self._log(f"[{stage}] candidate {i + 1}/{len(codes)}: fitness={fit_str}"
                      + (f"  err={err[:120]}" if err else ""))

        # Pick the highest-fitness valid candidate.
        best_idx: Optional[int] = None
        best_fit: Optional[float] = None
        for i, c in enumerate(rec.candidates):
            if c.fitness is None:
                continue
            if best_fit is None or c.fitness > best_fit:
                best_fit = c.fitness
                best_idx = i
        if best_idx is not None:
            rec.chosen_index = best_idx
            rec.chosen_fitness = best_fit
            rec.chosen_code = rec.candidates[best_idx].code
            self._log(f"[{stage}] selected candidate #{best_idx + 1}  "
                      f"fitness={best_fit:.4f}")
        else:
            self._log(f"[{stage}] all candidates failed; skipping stage.")
        return rec

    # ------------------------------------------------------------------
    def run(self) -> SPIORunResult:
        # Initial sandbox observation (used by the plan prompt + by
        # ``_observe_prefix`` when no prefix is selected yet).
        self.sandbox.reset()
        init_obs = self.sandbox._observe().text

        plan = self.generate_plan(init_obs)
        if self.verbose:
            for stage in STAGES:
                preview = plan.get(stage, "")[:160].replace("\n", " | ")
                self._log(f"[plan] {stage}: {preview}")

        chosen_prefix: list[PipelineStep] = []
        stage_records: list[StageRecord] = []
        best_fit: Optional[float] = None

        for stage in CODE_STAGES:
            stage_plan = plan.get(stage, "(no plan provided)")
            rec = self._select_stage(
                stage=stage, stage_plan=stage_plan,
                chosen_prefix=chosen_prefix,
            )
            stage_records.append(rec)
            if rec.chosen_code is not None:
                chosen_prefix.append(self._wrap_candidate(rec.chosen_code))
                if rec.chosen_fitness is not None:
                    if best_fit is None or rec.chosen_fitness > best_fit:
                        best_fit = rec.chosen_fitness

        return SPIORunResult(
            nl_plan=plan,
            chosen_prefix=chosen_prefix,
            stage_records=stage_records,
            best_fitness_in_loop=best_fit,
        )


__all__ = [
    "SPIOAgent",
    "SPIORunResult",
    "StageRecord",
    "CandidateRecord",
]
