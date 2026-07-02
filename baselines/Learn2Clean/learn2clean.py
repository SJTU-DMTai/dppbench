"""Top-level Learn2Clean orchestrator.

Public entry: :class:`Learn2Clean`.run() returns a dict whose schema is aligned
with :class:`baselines.CtxPipe.ctxpipe.CtxPipe`.run() so the Learn2Clean
baseline plugs into existing benchmark harnesses without changes.
"""
from __future__ import annotations

import json
import os
import random as _random
import time
import warnings
from typing import Optional, Tuple

import numpy as np

from baselines.common.config import (
    default_config_path,
    load_baseline_config,
    resolve_config_value,
)
from baselines.CtxPipe.evaluator import CtxPipeEvaluator
from baselines.common.pipeline import DataContext, Pipeline
from baselines.common.pipeline_constraints import is_legal
from baselines.common.context import _infer_rec_context, _infer_tabular_context

from .agent import TabularQAgent
from .env import Learn2CleanEnv
from .operator_catalog import CATALOG  # noqa: F401  (also triggers SAGA sync)
from .trainer import Learn2CleanTrainer


CONFIG_KEYS = (
    "n_episodes",
    "max_steps",
    "small_n",
    "eval_full",
    "gamma",
    "lr",
    "temperature_init",
    "temperature_final",
    "reward_max",
    "illegal_reward",
    "improvement_eps",
    "seed",
    "fast_train",
)


class Learn2Clean:
    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir: Optional[str] = None,
        n_episodes: Optional[int] = None,
        max_steps: Optional[int] = None,
        small_n: Optional[int] = None,
        eval_full: Optional[bool] = None,
        gamma: Optional[float] = None,
        lr: Optional[float] = None,
        temperature_init: Optional[float] = None,
        temperature_final: Optional[float] = None,
        reward_max: Optional[float] = None,
        illegal_reward: Optional[float] = None,
        improvement_eps: Optional[float] = None,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True,
        device: str = "cpu",
        fast_train: Optional[bool] = None,
        config_path: Optional[str] = None,
    ) -> None:
        cfg = load_baseline_config(
            config_path or default_config_path(__file__), CONFIG_KEYS
        )
        n_episodes = resolve_config_value(cfg, "n_episodes", n_episodes)
        max_steps = resolve_config_value(cfg, "max_steps", max_steps)
        small_n = resolve_config_value(cfg, "small_n", small_n)
        eval_full = resolve_config_value(cfg, "eval_full", eval_full)
        gamma = resolve_config_value(cfg, "gamma", gamma)
        lr = resolve_config_value(cfg, "lr", lr)
        temperature_init = resolve_config_value(
            cfg, "temperature_init", temperature_init
        )
        temperature_final = resolve_config_value(
            cfg, "temperature_final", temperature_final
        )
        reward_max = resolve_config_value(cfg, "reward_max", reward_max)
        illegal_reward = resolve_config_value(cfg, "illegal_reward", illegal_reward)
        improvement_eps = resolve_config_value(
            cfg, "improvement_eps", improvement_eps
        )
        seed = resolve_config_value(cfg, "seed", seed)
        fast_train = resolve_config_value(cfg, "fast_train", fast_train)

        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir
        self.n_episodes = int(n_episodes)
        self.max_steps = int(max_steps)
        self.small_n = int(small_n) if small_n else 0
        self.eval_full = bool(eval_full)
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.temperature_init = float(temperature_init)
        self.temperature_final = float(temperature_final)
        self.reward_max = float(reward_max)
        self.illegal_reward = float(illegal_reward)
        self.improvement_eps = float(improvement_eps)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.device = device
        self.fast_train = bool(fast_train)

        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "Learn2Clean", data_name
        )
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _build_context(self, evaluator: CtxPipeEvaluator) -> Tuple[DataContext, object]:
        data = evaluator._executor._load_data()
        if evaluator.task_type == "rec":
            ctx = _infer_rec_context(self.data_name, {}, data)
        else:
            ctx = _infer_tabular_context(self.data_name, {}, data)
        return ctx, data

    # ------------------------------------------------------------------
    def run(self) -> dict:
        np.random.seed(self.seed)
        _random.seed(self.seed)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        t0 = time.time()

        train_evaluator = CtxPipeEvaluator(
            task_dir=self.task_dir,
            data_name=self.data_name,
            data_dir=self.data_dir,
            verbose=self.verbose,
            small_n=self.small_n,
            seed=self.seed,
            device=self.device,
        )
        if self.fast_train:
            train_evaluator.set_fast_mode(True)
        ctx, _data = self._build_context(train_evaluator)

        if self.verbose:
            print("=" * 60)
            print(f"[Learn2Clean] dataset={self.data_name}  task={ctx.task_type}")
            print(f"[Learn2Clean] numeric={len(ctx.numeric_cols)}  "
                  f"categorical={len(ctx.categorical_cols)}  "
                  f"target={ctx.target_col}")
            print(f"[Learn2Clean] n_episodes={self.n_episodes}  "
                  f"max_steps={self.max_steps}  small_n={self.small_n or 'OFF'}  "
                  f"gamma={self.gamma}  lr={self.lr}  "
                  f"T={self.temperature_init}->{self.temperature_final}")
            print("=" * 60)

        env = Learn2CleanEnv(
            evaluator=train_evaluator,
            ctx=ctx,
            max_steps=self.max_steps,
            reward_max=self.reward_max,
            illegal_reward=self.illegal_reward,
            improvement_eps=self.improvement_eps,
            seed=self.seed,
        )
        agent = TabularQAgent(
            n_actions=env.n_actions,
            gamma=self.gamma,
            lr=self.lr,
            temperature_init=self.temperature_init,
            temperature_final=self.temperature_final,
            seed=self.seed,
        )
        trainer = Learn2CleanTrainer(
            env=env,
            agent=agent,
            n_episodes=self.n_episodes,
            verbose=self.verbose,
        )

        result = trainer.train()
        best_pipe = result.best_pipeline
        best_fitness: Optional[float] = float(result.best_fitness) if (
            result.best_fitness is not None and not np.isnan(result.best_fitness)
        ) else None
        best_metrics = dict(result.best_metrics or {})
        eval_error: Optional[str] = None

        # Persist artefacts.
        out_yaml_path = os.path.join(self.output_dir, "best_pipeline.yaml")
        with open(out_yaml_path, "w", encoding="utf-8") as f:
            f.write(best_pipe.to_yaml())

        history_payload = [
            {
                "episode": rec.episode,
                "reward": rec.reward,
                "success": rec.success,
                "fitness": rec.fitness,
                "metrics": rec.metrics,
                "ops": rec.ops,
                "n_steps": rec.n_steps,
                "temperature": rec.temperature,
                "duration_seconds": rec.duration_seconds,
            }
            for rec in result.history
        ]
        history_path = os.path.join(self.output_dir, "train_history.json")
        try:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history_payload, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"[Learn2Clean] warning: failed to save history: {e}")

        q_table_path = os.path.join(self.output_dir, "q_table.json")
        try:
            agent.save(q_table_path)
        except Exception as e:
            if self.verbose:
                print(f"[Learn2Clean] warning: failed to save q_table: {e}")
            q_table_path = None

        for i, sp in enumerate(result.top_k):
            path = os.path.join(self.output_dir, f"top{i+1}_pipeline.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write(sp["yaml"])

        # Final full-data evaluation.
        if self.eval_full:
            try:
                full_eval = CtxPipeEvaluator(
                    task_dir=self.task_dir,
                    data_name=self.data_name,
                    data_dir=self.data_dir,
                    verbose=self.verbose,
                    small_n=0,
                    seed=self.seed,
                    device=self.device,
                )
                ev = full_eval.evaluate(best_pipe)
                if ev.success and ev.fitness is not None:
                    best_fitness = float(ev.fitness)
                    best_metrics = dict(ev.metrics or {})
                else:
                    eval_error = ev.error
                    if self.verbose:
                        print(f"[Learn2Clean] full eval failed: {ev.error}")
            except Exception as e:
                eval_error = f"{type(e).__name__}: {e}"
                if self.verbose:
                    print(f"[Learn2Clean] full eval raised: {e}")

        duration = time.time() - t0

        if self.verbose:
            print("=" * 60)
            print(f"[Learn2Clean] DONE in {duration:.1f}s")
            fit_str = f"{best_fitness:.4f}" if isinstance(best_fitness, float) else "n/a"
            print(f"[Learn2Clean] best fitness = {fit_str}")
            print(f"[Learn2Clean] best metrics = {best_metrics}")
            print(f"[Learn2Clean] best ops     = {best_pipe.op_names()}")
            print(f"[Learn2Clean] best yaml    -> {out_yaml_path}")
            print(f"[Learn2Clean] history      -> {history_path}")
            print(f"[Learn2Clean] q_table      -> {q_table_path}")
            print(f"[Learn2Clean] q_states     = {agent.n_states}")
            print(f"[Learn2Clean] unique evals = {train_evaluator.n_unique_evaluations}")
            print("=" * 60)

        return {
            "best_pipeline_yaml": best_pipe.to_yaml(),
            "best_pipeline_path": out_yaml_path,
            "best_fitness": best_fitness,
            "best_metrics": best_metrics,
            "eval_error": eval_error,
            "is_legal": is_legal(best_pipe, ctx.task_type),
            "final_pipeline_ops": best_pipe.op_names(),
            "search_history": history_payload,
            "n_unique_evaluations": train_evaluator.n_unique_evaluations,
            "duration_seconds": duration,
            "output_dir": self.output_dir,
            "q_table_path": q_table_path,
            "top_k": list(result.top_k),
        }


__all__ = ["Learn2Clean"]
