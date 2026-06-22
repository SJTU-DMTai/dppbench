"""Top-level ReAct orchestrator.

Mirrors the structure of :class:`baselines.DataMaster.data_master.DataMaster`
(infer context -> build sandbox/evaluator/llm -> run agent -> repair ->
final downstream eval -> persist artefacts) so the CLI surface stays
consistent across the dppbench baselines.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from baselines.common.config import (
    default_config_path,
    load_baseline_config,
    resolve_config_value,
)
from baselines.SAGA.pipeline import DataContext, Pipeline
from baselines.SAGA.pipeline_constraints import is_legal, repair
from baselines.SAGA.saga import _infer_rec_context, _infer_tabular_context
from baselines.DeepPrep.llm_client import LLMClient
from baselines.DeepPrep.sandbox import Sandbox

from .agent import ReActAgent
from .evaluator import ReActEvaluator


# ---------------------------------------------------------------------------
# YAML example for the prompt: load amazon_beauty's reference pipeline if it
# exists; otherwise fall back to a small in-memory snippet.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DEFAULT_YAML_EXAMPLE_PATH = os.path.join(
    _REPO_ROOT, "dppbench", "tasks", "amazon_beauty", "pre_process.yaml"
)
_FALLBACK_YAML_EXAMPLE = """\
pipeline:
  - op: HandleMV
    target: both
    params: {}
  - op: LabelEncode
    target: both
    params: {}
"""


def _load_yaml_example() -> str:
    if os.path.isfile(_DEFAULT_YAML_EXAMPLE_PATH):
        try:
            with open(_DEFAULT_YAML_EXAMPLE_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    return _FALLBACK_YAML_EXAMPLE


CONFIG_KEYS = (
    "llm_backend",
    "llm_model",
    "api_key",
    "base_url",
    "temperature",
    "max_tokens",
    "timeout",
    "max_turns",
    "max_retry_per_turn",
    "max_err_cnt",
    "small_n",
    "eval_full",
    "downstream_eval_n",
    "seed",
    "fast_train",
)


class ReAct:
    """ReAct baseline runner: full-pipeline YAML per turn + best-of-N."""

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
        # ---- ReAct loop ----
        max_turns: Optional[int] = None,
        max_retry_per_turn: Optional[int] = None,
        max_err_cnt: Optional[int] = None,
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
        max_turns = resolve_config_value(cfg, "max_turns", max_turns)
        max_retry_per_turn = resolve_config_value(
            cfg, "max_retry_per_turn", max_retry_per_turn
        )
        max_err_cnt = resolve_config_value(cfg, "max_err_cnt", max_err_cnt)
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

        self.max_turns = int(max_turns)
        self.max_retry_per_turn = int(max_retry_per_turn)
        self.max_err_cnt = int(max_err_cnt)

        self.small_n = int(small_n) if small_n else 0
        self.eval_full = bool(eval_full)
        self.downstream_eval_n = int(downstream_eval_n) if downstream_eval_n else 0
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.device = device
        self.fast_train = bool(fast_train)

        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "ReAct", data_name
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

        agent_evaluator = ReActEvaluator(
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

        try:
            ctx = self._infer_context(sandbox)

            if self.verbose:
                print("=" * 60)
                print(f"[ReAct] dataset={self.data_name}  task={ctx.task_type}")
                print(f"[ReAct] numeric={len(ctx.numeric_cols)}  "
                      f"categorical={len(ctx.categorical_cols)}  "
                      f"list={len(ctx.list_cols)}  text={len(ctx.text_cols)}")
                print(f"[ReAct] target={ctx.target_col}  time={ctx.time_col}  "
                      f"id={ctx.id_col}  aux={ctx.aux_dfs}")
                print(f"[ReAct] llm={self.llm_backend}/{self.llm_model}  "
                      f"max_turns={self.max_turns}  "
                      f"max_retry_per_turn={self.max_retry_per_turn}  "
                      f"max_err_cnt={self.max_err_cnt}")
                print(f"[ReAct] small_n={self.small_n or 'OFF'}  "
                      f"eval_full={self.eval_full}  "
                      f"downstream_eval_n={self.downstream_eval_n or 'FULL'}")
                print("=" * 60)

            yaml_example = _load_yaml_example()
            llm = self._build_llm()
            agent = ReActAgent(
                llm=llm,
                sandbox=sandbox,
                ctx=ctx,
                downstream_evaluator=_agent_eval_fn,
                yaml_example=yaml_example,
                max_turns=self.max_turns,
                max_retry_per_turn=self.max_retry_per_turn,
                max_err_cnt=self.max_err_cnt,
                seed=self.seed,
                verbose=self.verbose,
            )
            run = agent.run()
            pipeline: Pipeline = run.best_pipeline

            # Final structural repair (safety net; the agent itself does not
            # repair so the LLM's mistakes stay visible in the trajectory).
            repair(pipeline, ctx.task_type, ctx)
            legal = is_legal(pipeline, ctx.task_type)

            if self.verbose:
                fit_str = (
                    f"{run.best_fitness:.4f}"
                    if isinstance(run.best_fitness, float)
                    else "n/a"
                )
                print(f"[ReAct] agent done. success={run.success}  "
                      f"turns={run.n_turns}  errors={run.n_errors}  "
                      f"best_turn={run.best_turn}  best_fitness={fit_str}  "
                      f"steps={[s.op for s in pipeline.steps]}  legal={legal}")
        finally:
            sandbox.cleanup()

        # Final downstream evaluation (NOT fed back to the agent).
        eval_metrics: dict = {}
        eval_fitness: Optional[float] = None
        eval_error: Optional[str] = None
        if self.eval_full:
            evaluator = ReActEvaluator(
                task_dir=self.task_dir,
                data_name=self.data_name,
                data_dir=self.data_dir,
                verbose=self.verbose,
                small_n=0,
                seed=self.seed,
                device=self.device,
            )
            try:
                ev = evaluator.evaluate(pipeline)
                eval_metrics = dict(ev.metrics or {})
                eval_fitness = ev.fitness if ev.success else None
                eval_error = ev.error
            except Exception as e:  # pragma: no cover
                eval_error = f"{type(e).__name__}: {e}"

        # Persist artefacts.
        out_yaml_path = os.path.join(self.output_dir, "best_pipeline.yaml")
        yaml_text = pipeline.to_yaml()
        with open(out_yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_text)

        log_path = os.path.join(self.output_dir, "agent_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "data_name": self.data_name,
                    "task_type": ctx.task_type,
                    "agent_success": run.success,
                    "n_turns": run.n_turns,
                    "n_errors": run.n_errors,
                    "best_turn": run.best_turn,
                    "best_fitness": run.best_fitness,
                    "best_metrics": run.best_metrics,
                    "max_turns": self.max_turns,
                    "max_retry_per_turn": self.max_retry_per_turn,
                    "max_err_cnt": self.max_err_cnt,
                    "transcript": run.transcript,
                    "ops": [s.op for s in pipeline.steps],
                    "downstream_fitness": eval_fitness,
                    "downstream_metrics": eval_metrics,
                    "downstream_error": eval_error,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        traj_path = os.path.join(self.output_dir, "trajectory.json")
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "turn": t.turn,
                        "thought": t.thought,
                        "pipeline_yaml": t.pipeline_yaml,
                        "parsed_ops": t.parsed_ops,
                        "pipeline_steps": t.pipeline_steps,
                        "status": t.status,
                        "error": t.error,
                        "fitness": t.fitness,
                        "metrics": t.metrics,
                        "obs_text": t.obs_text,
                        "is_terminate": t.is_terminate,
                    }
                    for t in run.trajectory
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )

        summary_path = os.path.join(self.output_dir, "run_summary.json")
        duration = time.time() - t0
        summary = {
            "data_name": self.data_name,
            "task_type": ctx.task_type,
            "agent_success": run.success,
            "best_turn": run.best_turn,
            "best_fitness_in_loop": run.best_fitness,
            "best_metrics_in_loop": run.best_metrics,
            "downstream_fitness": eval_fitness,
            "downstream_metrics": eval_metrics,
            "downstream_error": eval_error,
            "n_turns": run.n_turns,
            "n_errors": run.n_errors,
            "is_legal": legal,
            "final_pipeline_ops": [s.op for s in pipeline.steps],
            "duration_seconds": duration,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print("=" * 60)
            print(f"[ReAct] DONE in {duration:.1f}s")
            fit_str = f"{eval_fitness:.4f}" if isinstance(eval_fitness, float) else "n/a"
            print(f"[ReAct] downstream fitness = {fit_str}")
            print(f"[ReAct] downstream metrics = {eval_metrics}")
            if eval_error:
                print(f"[ReAct] eval error: {eval_error}")
            print(f"[ReAct] best pipeline saved to: {out_yaml_path}")
            print(f"[ReAct] agent log saved to:    {log_path}")
            print(f"[ReAct] trajectory saved to:   {traj_path}")
            print(f"[ReAct] run summary saved to:  {summary_path}")
            print("=" * 60)

        return {
            "best_pipeline_yaml": yaml_text,
            "best_pipeline_path": out_yaml_path,
            "best_fitness": eval_fitness,
            "best_metrics": eval_metrics,
            "best_fitness_in_loop": run.best_fitness,
            "best_metrics_in_loop": run.best_metrics,
            "eval_error": eval_error,
            "agent_success": run.success,
            "agent_n_turns": run.n_turns,
            "agent_n_errors": run.n_errors,
            "agent_best_turn": run.best_turn,
            "final_pipeline_ops": [s.op for s in pipeline.steps],
            "is_legal": legal,
            "task_type": ctx.task_type,
            "duration_seconds": duration,
            "output_dir": self.output_dir,
            "agent_log_path": log_path,
            "trajectory_path": traj_path,
            "run_summary_path": summary_path,
        }


__all__ = ["ReAct"]
