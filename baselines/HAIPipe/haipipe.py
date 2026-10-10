"""HAIPipe baseline trained independently for each task/model pair."""
from __future__ import annotations

import json
import os
import random
import time
from typing import Optional


from baselines.common.config import default_config_path, load_baseline_config, resolve_config_value
from baselines.common.pipeline import check_pipeline
from baselines.common.search_evaluator import SearchPipelineEvaluator
from baselines.common.utils import get_data_ctx

from .active_learning import ActiveCandidateSelector, ScoreNN
from .combiner import combine_pipelines, select_best_candidate
from .env import HAIPipeEnv
from .hi_pipeline import build_hi_pipeline
from .pretrain import CrossTaskPretrainer, save_alternative_evaluator


CONFIG_KEYS = (
    "max_pipeline_len", "small_n", "eval_full", "seed", "fast_train",
    "hi_source", "al_k", "al_iterations",
    "learning_rate", "buffer_capacity", "gamma", "batch_size", "logic_batch_size",
    "epsilon_start", "epsilon_end", "epsilon_decay", "train_interval", "frames",
    "al_epochs", "al_learning_rate", "al_batch_size",
)


class HAIPipe:
    def __init__(
        self, task_dir: str, data_name: str, data_dir: Optional[str] = None,
        max_pipeline_len: Optional[int] = None, small_n: Optional[int] = None,
        eval_full: Optional[bool] = None, seed: Optional[int] = None,
        output_dir: Optional[str] = None, verbose: bool = True, device: str = "cpu",
        fast_train: Optional[bool] = None, hi_source: Optional[str] = None,
        model_name: Optional[str] = None, config_path: Optional[str] = None,
        al_k: Optional[int] = None, al_iterations: Optional[int] = None,
        **_legacy,
    ) -> None:
        cfg = load_baseline_config(config_path or default_config_path(__file__), CONFIG_KEYS)
        value = lambda key, explicit: resolve_config_value(cfg, key, explicit)
        self.task_dir, self.data_name, self.data_dir = task_dir, data_name, data_dir
        self.max_pipeline_len = int(value("max_pipeline_len", max_pipeline_len))
        self.small_n = value("small_n", small_n)
        self.eval_full = bool(value("eval_full", eval_full))
        self.seed = int(value("seed", seed))
        self.fast_train = bool(value("fast_train", fast_train))
        self.hi_source = str(value("hi_source", hi_source))
        self.al_k = int(value("al_k", al_k))
        self.al_iterations = int(value("al_iterations", al_iterations))
        self.learning_rate = float(cfg.get("learning_rate", 1e-5))
        self.buffer_capacity = int(cfg.get("buffer_capacity", 2000))
        self.gamma = float(cfg.get("gamma", 1.0))
        self.batch_size = int(cfg.get("batch_size", 100))
        self.logic_batch_size = int(cfg.get("logic_batch_size", 20))
        self.epsilon_start = float(cfg.get("epsilon_start", 1.0))
        self.epsilon_end = float(cfg.get("epsilon_end", 0.4))
        self.epsilon_decay = float(cfg.get("epsilon_decay", 500.0))
        self.train_interval = int(cfg.get("train_interval", 100))
        self.frames = int(cfg.get("frames", 56000))
        self.model_name, self.verbose, self.device = model_name, bool(verbose), device
        self.output_dir = output_dir or os.path.join("outputs", "HAIPipe", data_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> dict:
        started = time.time()
        evaluator = SearchPipelineEvaluator(
            task_dir=self.task_dir, data_name=self.data_name, data_dir=self.data_dir,
            small_n=self.small_n, seed=self.seed, verbose=self.verbose,
            device=self.device, model_name=self.model_name,
        )
        evaluator.set_evaluate_std_test(False)
        if self.fast_train:
            evaluator.set_fast_mode(True)
        loaded = evaluator.load_data()
        ctx = get_data_ctx(self.data_name, loaded, evaluator.task_type)
        hi = build_hi_pipeline(
            ctx, random.Random(self.seed), task_dir=self.task_dir,
            max_len=self.max_pipeline_len, hi_source=self.hi_source,
        )
        env = HAIPipeEnv(evaluator, ctx, model_name=self.model_name, seed=self.seed)
        aipipe_path = os.path.join(self.output_dir, "aipipe.pt")
        policy_trainer = CrossTaskPretrainer(
            [env], frames=self.frames, learning_rate=self.learning_rate,
            buffer_capacity=self.buffer_capacity, batch_size=self.batch_size,
            gamma=self.gamma, train_interval=self.train_interval,
            logic_batch_size=self.logic_batch_size,
            epsilon_start=self.epsilon_start, epsilon_end=self.epsilon_end,
            epsilon_decay=self.epsilon_decay, seed=self.seed, device=self.device,
        )
        policy_training = policy_trainer.train(
            aipipe_path, training_mode="target_task_online",
        )
        ai, trace = env.rollout(policy_trainer.agent, epsilon=0.0)
        candidates = combine_pipelines(
            hi, ai, ctx, random.Random(self.seed), max_len=self.max_pipeline_len,
        )
        al_model = ScoreNN(env.data_dim, len(env.primitive_vocabulary))
        selector = ActiveCandidateSelector(
            al_model, k_total=self.al_k, iterations=self.al_iterations, seed=self.seed,
        )
        selected = select_best_candidate(
            evaluator, candidates, selector=selector,
            dataset_feature=env.data_feature, vocabulary=env.primitive_vocabulary,
        )
        al_path = os.path.join(self.output_dir, "alternative_evaluator.pt")
        save_alternative_evaluator(
            al_model, al_path, data_dim=env.data_dim,
            primitive_vocabulary=env.primitive_vocabulary,
            provenance={
                "training_mode": "target_task_online",
                "training_tasks": [self.data_name],
                "model_name": self.model_name,
                "active_learning_evaluations": len(selected.candidates),
            },
        )
        output = selected.pipeline
        full_fitness, full_metrics = None, {}
        if self.eval_full:
            evaluator.set_small_n(0)
            if self.fast_train:
                evaluator.set_fast_mode(False)
            final = evaluator.evaluate(output)
            if final.success:
                full_fitness, full_metrics = float(final.fitness), dict(final.metrics or {})
        report = check_pipeline(output, ctx)
        yaml_path = os.path.join(self.output_dir, "best_pipeline.yaml")
        log_path = os.path.join(self.output_dir, "haipipe_log.json")
        with open(yaml_path, "w", encoding="utf-8") as handle:
            handle.write(output.to_yaml())
        log = {
            "data_name": self.data_name, "hi_source": self.hi_source,
            "hi_ops": hi.op_names(), "ai_ops": ai.op_names(),
            "best_pipeline_yaml": output.to_yaml(), "best_pipeline_path": yaml_path,
            "best_fitness": selected.fitness, "best_metrics": selected.metrics,
            "best_full_fitness": full_fitness, "best_full_metrics": full_metrics,
            "is_legal": report.valid, "check_errors": report.errors,
            "aipipe_trace": trace, "active_learning": selected.candidates,
            "n_hai_candidates": len(candidates),
            "n_unique_evaluations": evaluator.n_unique_evaluations,
            "duration_seconds": time.time() - started,
            "training_mode": "per_task_model",
            "policy_training": policy_training,
            "checkpoints": {"aipipe": aipipe_path, "alternative_evaluator": al_path},
        }
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(log, handle, indent=2)
        return log


__all__ = ["HAIPipe"]
