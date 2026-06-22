"""Top-level SPIO orchestrator.

Mirrors :class:`baselines.ReAct.react.ReAct`'s structure (infer context ->
build sandbox / evaluator / llm -> run agent -> structural scaffold ->
final downstream eval -> persist artefacts) so the CLI surface stays
consistent across the dppbench baselines.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Optional

from baselines.common.config import (
    default_config_path,
    load_baseline_config,
    resolve_config_value,
)
from baselines.DeepPrep.llm_client import LLMClient
from baselines.DeepPrep.sandbox import Sandbox
from baselines.SAGA.pipeline import DataContext, Pipeline, PipelineStep
from baselines.SAGA.pipeline_constraints import is_legal, repair
from baselines.SAGA.saga import _infer_rec_context, _infer_tabular_context

from .agent import SPIOAgent, SPIORunResult
from .evaluator import SPIOEvaluator
from .stages import build_prefix_scaffolded_pipeline, build_scaffolded_pipeline


CONFIG_KEYS = (
    "llm_backend",
    "llm_model",
    "api_key",
    "base_url",
    "temperature",
    "max_tokens",
    "timeout",
    "n_candidates",
    "max_retry",
    "stage_max_per_cat",
    "small_n",
    "eval_full",
    "downstream_eval_n",
    "seed",
    "fast_train",
)


class SPIO:
    """SPIO baseline runner: NL plan + per-stage best-of-N CustomOp selection."""

    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir: Optional[str] = None,
        # ---- LLM ----
        llm_backend: Optional[str] = None,
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        # ---- SPIO loop ----
        n_candidates: Optional[int] = None,
        max_retry: Optional[int] = None,
        stage_max_per_cat: Optional[int] = None,
        # ---- Eval ----
        small_n: Optional[int] = None,
        eval_full: Optional[bool] = None,
        downstream_eval_n: Optional[int] = None,
        seed: Optional[int] = None,
        # ---- IO ----
        output_dir: Optional[str] = None,
        verbose: bool = True,
        device: str = "cpu",
        fast_train: Optional[bool] = None,
        config_path: Optional[str] = None,
    ) -> None:
        cfg = load_baseline_config(
            config_path or default_config_path(__file__), CONFIG_KEYS
        )
        llm_backend = resolve_config_value(cfg, "llm_backend", llm_backend)
        llm_model = resolve_config_value(cfg, "llm_model", llm_model)
        api_key = resolve_config_value(cfg, "api_key", api_key)
        base_url = resolve_config_value(cfg, "base_url", base_url)
        temperature = resolve_config_value(cfg, "temperature", temperature)
        max_tokens = resolve_config_value(cfg, "max_tokens", max_tokens)
        timeout = resolve_config_value(cfg, "timeout", timeout)
        n_candidates = resolve_config_value(cfg, "n_candidates", n_candidates)
        max_retry = resolve_config_value(cfg, "max_retry", max_retry)
        stage_max_per_cat = resolve_config_value(
            cfg, "stage_max_per_cat", stage_max_per_cat
        )
        small_n = resolve_config_value(cfg, "small_n", small_n)
        eval_full = resolve_config_value(cfg, "eval_full", eval_full)
        downstream_eval_n = resolve_config_value(
            cfg, "downstream_eval_n", downstream_eval_n
        )
        seed = resolve_config_value(cfg, "seed", seed)
        fast_train = resolve_config_value(cfg, "fast_train", fast_train)

        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir

        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)

        self.n_candidates = int(n_candidates)
        self.max_retry = int(max_retry)
        self.stage_max_per_cat = int(stage_max_per_cat)

        self.small_n = int(small_n) if small_n else 0
        self.eval_full = bool(eval_full)
        self.downstream_eval_n = int(downstream_eval_n) if downstream_eval_n else 0
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.device = device
        self.fast_train = bool(fast_train)

        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "SPIO", data_name
        )
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _build_llm(self) -> LLMClient:
        return LLMClient(
            backend=self.llm_backend,
            model=self.llm_model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

    def _infer_context(self, sandbox: Sandbox) -> DataContext:
        if sandbox.data is None:
            sandbox.reset()
        if sandbox.task_type == "rec":
            return _infer_rec_context(self.data_name, {}, sandbox.data)
        return _infer_tabular_context(self.data_name, {}, sandbox.data)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        t0 = time.time()

        agent_evaluator = SPIOEvaluator(
            task_dir=self.task_dir,
            data_name=self.data_name,
            data_dir=self.data_dir,
            small_n=self.downstream_eval_n,
            seed=self.seed,
            verbose=False,
            device=self.device,
        )
        if self.fast_train:
            agent_evaluator.set_fast_mode(True)

        def _agent_eval_fn(pipeline: Pipeline):
            return agent_evaluator.evaluate_for_agent(pipeline)

        sandbox = Sandbox(
            task_dir=self.task_dir,
            data_name=self.data_name,
            data_dir=self.data_dir,
            small_n=self.small_n,
            seed=self.seed,
        )

        rng = random.Random(self.seed)

        run: Optional[SPIORunResult] = None
        ctx: Optional[DataContext] = None
        final_pipeline: Pipeline = Pipeline()
        legal = False
        repair_applied = False

        try:
            ctx = self._infer_context(sandbox)

            if self.verbose:
                print("=" * 60)
                print(f"[SPIO] dataset={self.data_name}  task={ctx.task_type}")
                print(f"[SPIO] numeric={len(ctx.numeric_cols)}  "
                      f"categorical={len(ctx.categorical_cols)}  "
                      f"list={len(ctx.list_cols)}  text={len(ctx.text_cols)}")
                print(f"[SPIO] target={ctx.target_col}  time={ctx.time_col}  "
                      f"id={ctx.id_col}  aux={ctx.aux_dfs}")
                print(f"[SPIO] llm={self.llm_backend}/{self.llm_model}  "
                      f"n_candidates={self.n_candidates}  "
                      f"max_retry={self.max_retry}")
                print(f"[SPIO] small_n={self.small_n or 'OFF'}  "
                      f"eval_full={self.eval_full}  "
                      f"downstream_eval_n={self.downstream_eval_n or 'FULL'}")
                print("=" * 60)

            def _scaffold_fn(steps: list[PipelineStep]) -> Pipeline:
                return build_scaffolded_pipeline(steps, ctx, rng)

            def _prefix_scaffold_fn(steps: list[PipelineStep]) -> Pipeline:
                return build_prefix_scaffolded_pipeline(steps, ctx, rng)

            llm = self._build_llm()
            agent = SPIOAgent(
                llm=llm,
                sandbox=sandbox,
                ctx=ctx,
                evaluate_fn=_agent_eval_fn,
                build_scaffold_fn=_scaffold_fn,
                build_prefix_scaffold_fn=_prefix_scaffold_fn,
                n_candidates=self.n_candidates,
                max_retry=self.max_retry,
                stage_max_per_cat=self.stage_max_per_cat,
                seed=self.seed,
                verbose=self.verbose,
            )
            run = agent.run()
            final_pipeline = _scaffold_fn(run.chosen_prefix)
            legal = is_legal(final_pipeline, ctx.task_type)
            if not legal:
                repair(final_pipeline, ctx.task_type, ctx)
                repair_applied = True
                legal = is_legal(final_pipeline, ctx.task_type)

            if self.verbose:
                in_loop = run.best_fitness_in_loop
                in_loop_str = (
                    f"{in_loop:.4f}" if isinstance(in_loop, float) else "n/a"
                )
                ops = [s.op for s in final_pipeline.steps]
                print(f"[SPIO] agent done. best_in_loop={in_loop_str}  "
                      f"chosen_code_steps={len(run.chosen_prefix)}  "
                      f"final_ops={ops}")
        finally:
            sandbox.cleanup()

        # ---- Final downstream evaluation (NOT fed back to the agent) ----
        eval_metrics: dict = {}
        eval_fitness: Optional[float] = None
        eval_error: Optional[str] = None
        legal = is_legal(final_pipeline, ctx.task_type) if ctx is not None else legal
        if self.eval_full:
            evaluator = SPIOEvaluator(
                task_dir=self.task_dir,
                data_name=self.data_name,
                data_dir=self.data_dir,
                verbose=self.verbose,
                small_n=0,
                seed=self.seed,
                device=self.device,
            )
            try:
                ev = evaluator.evaluate(final_pipeline)
                eval_metrics = dict(ev.metrics or {})
                eval_fitness = ev.fitness if ev.success else None
                eval_error = ev.error
            except Exception as e:  # pragma: no cover
                eval_error = f"{type(e).__name__}: {e}"

        # ---- Persist artefacts ----
        out_yaml_path = os.path.join(self.output_dir, "best_pipeline.yaml")
        yaml_text = final_pipeline.to_yaml()
        with open(out_yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_text)

        plan_path = os.path.join(self.output_dir, "nl_plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(run.nl_plan if run else {}, f, ensure_ascii=False, indent=2)

        traj_path = os.path.join(self.output_dir, "trajectory.json")
        traj_payload = []
        if run is not None:
            for sr in run.stage_records:
                traj_payload.append({
                    "stage": sr.stage,
                    "plan": sr.plan,
                    "obs_text": sr.obs_text,
                    "candidates": [
                        {
                            "fitness": c.fitness,
                            "metrics": c.metrics,
                            "error": c.error,
                            "code": c.code,
                        }
                        for c in sr.candidates
                    ],
                    "chosen_index": sr.chosen_index,
                    "chosen_fitness": sr.chosen_fitness,
                })
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(traj_payload, f, ensure_ascii=False, indent=2)

        summary_path = os.path.join(self.output_dir, "run_summary.json")
        duration = time.time() - t0
        per_stage_fitness = {}
        if run is not None:
            for sr in run.stage_records:
                per_stage_fitness[sr.stage] = sr.chosen_fitness
        summary = {
            "data_name": self.data_name,
            "task_type": ctx.task_type if ctx is not None else None,
            "best_fitness_in_loop": run.best_fitness_in_loop if run else None,
            "per_stage_chosen_fitness": per_stage_fitness,
            "downstream_fitness": eval_fitness,
            "downstream_metrics": eval_metrics,
            "downstream_error": eval_error,
            "is_legal": legal,
            "repair_applied": repair_applied,
            "n_candidates": self.n_candidates,
            "final_pipeline_ops": [s.op for s in final_pipeline.steps],
            "duration_seconds": duration,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print("=" * 60)
            print(f"[SPIO] DONE in {duration:.1f}s")
            fit_str = (
                f"{eval_fitness:.4f}" if isinstance(eval_fitness, float) else "n/a"
            )
            print(f"[SPIO] downstream fitness = {fit_str}")
            print(f"[SPIO] downstream metrics = {eval_metrics}")
            if eval_error:
                print(f"[SPIO] eval error: {eval_error}")
            print(f"[SPIO] best pipeline saved to: {out_yaml_path}")
            print(f"[SPIO] nl plan saved to:       {plan_path}")
            print(f"[SPIO] trajectory saved to:    {traj_path}")
            print(f"[SPIO] run summary saved to:   {summary_path}")
            print("=" * 60)

        return {
            "best_pipeline_yaml": yaml_text,
            "best_pipeline_path": out_yaml_path,
            "best_fitness": eval_fitness,
            "best_metrics": eval_metrics,
            "best_fitness_in_loop": run.best_fitness_in_loop if run else None,
            "per_stage_chosen_fitness": per_stage_fitness,
            "eval_error": eval_error,
            "is_legal": legal,
            "repair_applied": repair_applied,
            "final_pipeline_ops": [s.op for s in final_pipeline.steps],
            "task_type": ctx.task_type if ctx is not None else None,
            "duration_seconds": duration,
            "output_dir": self.output_dir,
            "nl_plan_path": plan_path,
            "trajectory_path": traj_path,
            "run_summary_path": summary_path,
        }


__all__ = ["SPIO"]
