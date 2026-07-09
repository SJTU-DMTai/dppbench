"""Top-level HAIPipe adapter for DPPBench.

The adapter keeps HAIPipe's method structure: build a HI scaffold, learn an
AI-pipeline by component-level RL, enumerate HAI combinations, and select the
best HAI-pipeline by downstream feedback.  Operator construction, legality,
DAG assignment, and evaluation all go through ``baselines.common``.
"""
from __future__ import annotations

import json
import os
import random as _random
import time
from typing import Optional

from baselines.CtxPipe.agent import DQNAgent
from baselines.common.config import default_config_path, load_baseline_config, resolve_config_value
from baselines.common.data_ctx import DataContext
from baselines.common.pipeline import Pipeline, check_pipeline, is_legal
from baselines.common.search_evaluator import SearchPipelineEvaluator
from baselines.common.utils import get_data_ctx

from .env import HAIPipeEnv
from .hi_pipeline import build_hi_pipeline
from .trainer import HAIPipeTrainer, greedy_rollout


CONFIG_KEYS = (
    "n_episodes",
    "max_ai_steps",
    "max_pipeline_len",
    "n_combinations",
    "small_n",
    "eval_full",
    "gamma",
    "eps_start",
    "eps_end",
    "eps_decay_episodes",
    "batch_size",
    "target_update_every",
    "min_buffer",
    "hidden_dim",
    "lr",
    "buffer_capacity",
    "failure_reward",
    "seed",
    "fast_train",
    "hi_source",
)


class HAIPipe:
    """HAIPipe baseline runner."""

    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir: Optional[str] = None,
        n_episodes: Optional[int] = None,
        max_ai_steps: Optional[int] = None,
        max_pipeline_len: Optional[int] = None,
        n_combinations: Optional[int] = None,
        small_n: Optional[int] = None,
        eval_full: Optional[bool] = None,
        gamma: Optional[float] = None,
        eps_start: Optional[float] = None,
        eps_end: Optional[float] = None,
        eps_decay_episodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        target_update_every: Optional[int] = None,
        min_buffer: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        lr: Optional[float] = None,
        buffer_capacity: Optional[int] = None,
        failure_reward: Optional[float] = None,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True,
        device: str = "cpu",
        fast_train: Optional[bool] = None,
        hi_source: Optional[str] = None,
        model_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        cfg = load_baseline_config(
            config_path or default_config_path(__file__), CONFIG_KEYS
        )
        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir
        self.n_episodes = int(resolve_config_value(cfg, "n_episodes", n_episodes))
        self.max_ai_steps = int(resolve_config_value(cfg, "max_ai_steps", max_ai_steps))
        self.max_pipeline_len = int(resolve_config_value(cfg, "max_pipeline_len", max_pipeline_len))
        self.n_combinations = int(resolve_config_value(cfg, "n_combinations", n_combinations))
        self.small_n = resolve_config_value(cfg, "small_n", small_n)
        self.eval_full = bool(resolve_config_value(cfg, "eval_full", eval_full))
        self.gamma = float(resolve_config_value(cfg, "gamma", gamma))
        self.eps_start = float(resolve_config_value(cfg, "eps_start", eps_start))
        self.eps_end = float(resolve_config_value(cfg, "eps_end", eps_end))
        self.eps_decay_episodes = resolve_config_value(
            cfg, "eps_decay_episodes", eps_decay_episodes
        )
        self.batch_size = int(resolve_config_value(cfg, "batch_size", batch_size))
        self.target_update_every = int(resolve_config_value(
            cfg, "target_update_every", target_update_every
        ))
        self.min_buffer = int(resolve_config_value(cfg, "min_buffer", min_buffer))
        self.hidden_dim = int(resolve_config_value(cfg, "hidden_dim", hidden_dim))
        self.lr = float(resolve_config_value(cfg, "lr", lr))
        self.buffer_capacity = int(resolve_config_value(
            cfg, "buffer_capacity", buffer_capacity
        ))
        self.failure_reward = float(resolve_config_value(
            cfg, "failure_reward", failure_reward
        ))
        self.seed = int(resolve_config_value(cfg, "seed", seed))
        self.fast_train = bool(resolve_config_value(cfg, "fast_train", fast_train))
        self.hi_source = str(resolve_config_value(cfg, "hi_source", hi_source))
        self.verbose = bool(verbose)
        self.device = device
        self.model_name = model_name
        self.output_dir = output_dir or os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "HAIPipe", data_name,
        ))
        os.makedirs(self.output_dir, exist_ok=True)

    def _build_context(self, evaluator: SearchPipelineEvaluator) -> DataContext:
        data = evaluator.load_data()
        return get_data_ctx(self.data_name, data, evaluator.task_type)

    def _evaluate_pipeline(self, evaluator: SearchPipelineEvaluator, pipe: Pipeline) -> tuple[float | None, dict, str | None]:
        try:
            ev = evaluator.evaluate(pipe)
        except Exception as exc:  # pragma: no cover - defensive safety net
            return None, {}, f"{type(exc).__name__}: {exc}"
        if not ev.success:
            return None, dict(ev.metrics or {}), ev.error
        return float(ev.fitness), dict(ev.metrics or {}), ev.error

    def _record_history(self, records) -> list[dict]:
        return [
            {
                "episode": r.episode,
                "reward": r.reward,
                "success": r.success,
                "fitness": r.fitness,
                "metrics": r.metrics,
                "ai_ops": r.ai_ops,
                "hai_ops": r.hai_ops,
                "n_steps": r.n_steps,
                "duration_seconds": r.duration_seconds,
                "trace": r.trace,
            }
            for r in records
        ]

    def run(self) -> dict:
        t0 = time.time()
        evaluator = SearchPipelineEvaluator(
            task_dir=self.task_dir,
            data_name=self.data_name,
            data_dir=self.data_dir,
            small_n=self.small_n,
            seed=self.seed,
            verbose=self.verbose,
            device=self.device,
            model_name=self.model_name,
        )
        evaluator.set_evaluate_std_test(False)
        if self.fast_train:
            evaluator.set_fast_mode(True)
        ctx = self._build_context(evaluator)
        if ctx.task_type not in ("tabular", "rec"):
            return {
                "best_pipeline_yaml": None,
                "unsupported_task_type": ctx.task_type,
                "error": f"HAIPipe supports tabular/rec tasks, got {ctx.task_type}",
            }

        rng = _random.Random(self.seed)
        hi_pipeline = build_hi_pipeline(
            ctx, rng, max_len=self.max_pipeline_len, hi_source=self.hi_source
        )

        if self.verbose:
            print("=" * 60)
            print(f"[HAIPipe] dataset={self.data_name} task={ctx.task_type}")
            print(f"[HAIPipe] HI scaffold ops={hi_pipeline.op_names()}")
            print(f"[HAIPipe] n_episodes={self.n_episodes} max_ai_steps={self.max_ai_steps} "
                  f"n_combinations={self.n_combinations}")
            print("=" * 60)

        env = HAIPipeEnv(
            evaluator=evaluator,
            ctx=ctx,
            hi_pipeline=hi_pipeline,
            max_ai_steps=self.max_ai_steps,
            max_pipeline_len=self.max_pipeline_len,
            n_combinations=self.n_combinations,
            failure_reward=self.failure_reward,
            seed=self.seed,
        )

        history: list[dict] = []
        train_out: dict = {"best_record": None, "best_pipeline_yaml": None}
        greedy_out: dict = {}

        if env.component_ids:
            agent = DQNAgent(
                state_dim=env.state_dim,
                n_actions=env.n_actions,
                component_ids=env.component_ids,
                hidden=self.hidden_dim,
                lr=self.lr,
                buffer_capacity=self.buffer_capacity,
                seed=self.seed,
                device=self.device,
                context_dim=env.context_dim,
            )
            trainer = HAIPipeTrainer(
                env=env,
                agent=agent,
                n_episodes=self.n_episodes,
                eps_start=self.eps_start,
                eps_end=self.eps_end,
                eps_decay_episodes=self.eps_decay_episodes,
                gamma=self.gamma,
                batch_size=self.batch_size,
                target_update_every=self.target_update_every,
                min_buffer=self.min_buffer,
                verbose=self.verbose,
            )
            train_out = trainer.train()
            history = self._record_history(train_out["history"])

            inference_env = HAIPipeEnv(
                evaluator=evaluator,
                ctx=ctx,
                hi_pipeline=hi_pipeline,
                max_ai_steps=self.max_ai_steps,
                max_pipeline_len=self.max_pipeline_len,
                n_combinations=self.n_combinations,
                failure_reward=self.failure_reward,
                seed=self.seed + 7,
            )
            greedy_out = greedy_rollout(agent, inference_env, verbose=self.verbose)
            output_pipe = greedy_out["pipeline"].copy()
            best_fitness = greedy_out.get("fitness")
            best_metrics = dict(greedy_out.get("metrics") or {})
            eval_error = greedy_out.get("error")

            train_record = train_out.get("best_record")
            train_yaml = train_out.get("best_pipeline_yaml")
            train_fit = getattr(train_record, "fitness", None)
            if train_yaml and train_fit is not None:
                if best_fitness is None or train_fit > best_fitness:
                    output_pipe = Pipeline.from_yaml(train_yaml)
                    best_fitness = float(train_fit)
                    best_metrics = dict(getattr(train_record, "metrics", {}) or {})
                    eval_error = None
        else:
            output_pipe = hi_pipeline.copy()
            best_fitness, best_metrics, eval_error = self._evaluate_pipeline(evaluator, output_pipe)
            if self.verbose:
                print("[HAIPipe] no applicable AI components; using HI scaffold only")

        full_fitness = None
        full_metrics: dict = {}
        if self.eval_full:
            evaluator.set_small_n(0)
            if self.fast_train:
                evaluator.set_fast_mode(False)
            full_fitness, full_metrics, full_error = self._evaluate_pipeline(evaluator, output_pipe)
            eval_error = full_error
            if full_fitness is not None:
                best_fitness = full_fitness
                best_metrics = full_metrics

        check_report = check_pipeline(output_pipe, ctx)
        out_yaml = os.path.join(self.output_dir, "best_pipeline.yaml")
        out_log = os.path.join(self.output_dir, "haipipe_log.json")
        with open(out_yaml, "w", encoding="utf-8") as f:
            f.write(output_pipe.to_yaml())

        log = {
            "data_name": self.data_name,
            "task_type": ctx.task_type,
            "hi_source": self.hi_source,
            "hi_ops": hi_pipeline.op_names(),
            "best_pipeline_yaml": output_pipe.to_yaml(),
            "best_pipeline_path": out_yaml,
            "best_fitness": best_fitness,
            "best_metrics": best_metrics,
            "best_full_fitness": full_fitness,
            "best_full_metrics": full_metrics,
            "eval_error": eval_error,
            "is_legal": check_report.valid and is_legal(output_pipe, ctx.task_type),
            "check_errors": check_report.errors,
            "check_warnings": check_report.warnings,
            "final_pipeline_ops": output_pipe.op_names(),
            "greedy": {
                "fitness": greedy_out.get("fitness"),
                "ai_ops": greedy_out.get("ai_ops"),
                "hai_ops": greedy_out.get("hai_ops"),
                "trace": greedy_out.get("trace"),
                "candidates": greedy_out.get("candidates"),
            } if greedy_out else {},
            "history": history,
            "n_unique_evaluations": evaluator.n_unique_evaluations,
            "duration_seconds": time.time() - t0,
            "config": {
                "n_episodes": self.n_episodes,
                "max_ai_steps": self.max_ai_steps,
                "max_pipeline_len": self.max_pipeline_len,
                "n_combinations": self.n_combinations,
                "small_n": self.small_n,
                "eval_full": self.eval_full,
                "seed": self.seed,
                "fast_train": self.fast_train,
            },
        }
        with open(out_log, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

        if self.verbose:
            print("=" * 60)
            print(f"[HAIPipe] DONE in {time.time()-t0:.1f}s")
            print(f"[HAIPipe] best_fitness={best_fitness}")
            print(f"[HAIPipe] final ops={output_pipe.op_names()}")
            print(f"[HAIPipe] saved best pipeline -> {out_yaml}")
            print(f"[HAIPipe] saved log           -> {out_log}")
            print("=" * 60)
        return log


__all__ = ["HAIPipe"]
