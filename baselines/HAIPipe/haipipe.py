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
from baselines.common.training_utils import (validate_mode, pretrained_path, contract, load_pretrained, freeze_agent, resolved_model, isolate_training_data)

from .active_learning import ActiveCandidateSelector, ScoreNN
from .combiner import combine_pipelines, select_best_candidate
from .env import HAIPipeEnv
from .hi_pipeline import build_hi_pipeline
from .pretrain import CrossTaskPretrainer, save_alternative_evaluator


CONFIG_KEYS = (
    "training_mode", "pretrained_dir", "pretrained_checkpoint", "finetune_frames", "finetune_epochs",
    "finetune_eps_start", "finetune_eps_end",
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
        training_mode: Optional[str] = None, pretrained_dir: Optional[str] = None,
        pretrained_checkpoint: Optional[str] = None, finetune_frames: Optional[int] = None,
        finetune_epochs: Optional[int] = None,
        **_legacy,
    ) -> None:
        cfg = load_baseline_config(config_path or default_config_path(__file__), CONFIG_KEYS)
        value = lambda key, explicit: resolve_config_value(cfg, key, explicit)
        self.training_mode = validate_mode(training_mode or cfg["training_mode"])
        self.pretrained_dir = pretrained_dir or os.environ.get("DPPBENCH_PRETRAINED_DIR") or cfg["pretrained_dir"]
        self.pretrained_checkpoint = pretrained_checkpoint or cfg["pretrained_checkpoint"]
        self.finetune_frames = int(cfg["finetune_frames"] if finetune_frames is None else finetune_frames)
        self.finetune_epochs = int(cfg["finetune_epochs"] if finetune_epochs is None else finetune_epochs)
        self.finetune_eps_start = float(cfg["finetune_eps_start"])
        self.finetune_eps_end = float(cfg["finetune_eps_end"])
        if self.finetune_frames <= 0 or self.finetune_epochs <= 0:
            raise ValueError("Fine-tuning budgets must be positive")
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
        isolate_training_data(evaluator)
        if self.fast_train:
            evaluator.set_fast_mode(True)
        self.model_name = resolved_model(evaluator)
        loaded = evaluator.load_data()
        ctx = get_data_ctx(self.data_name, loaded, evaluator.task_type)
        hi = build_hi_pipeline(
            ctx, random.Random(self.seed), task_dir=self.task_dir,
            max_len=self.max_pipeline_len, hi_source=self.hi_source,
        )
        env = HAIPipeEnv(evaluator, ctx, model_name=self.model_name, seed=self.seed)
        aipipe_path = os.path.join(self.output_dir, "aipipe.pt")
        trainer = CrossTaskPretrainer(
            [env], frames=self.finetune_frames if self.training_mode == "finetune" else self.frames, learning_rate=self.learning_rate,
            buffer_capacity=self.buffer_capacity, batch_size=self.batch_size,
            gamma=self.gamma, train_interval=self.train_interval,
            logic_batch_size=self.logic_batch_size,
            epsilon_start=self.finetune_eps_start if self.training_mode == "finetune" else self.epsilon_start,
            epsilon_end=self.finetune_eps_end if self.training_mode == "finetune" else self.epsilon_end,
            epsilon_decay=self.epsilon_decay, seed=self.seed, device=self.device,
        )
        model_name = resolved_model(evaluator)
        agent = trainer.agent
        agent.checkpoint_metadata = contract("HAIPipe", env, model_name)
        source_path = pretrained_path("HAIPipe", model_name, self.pretrained_checkpoint, self.pretrained_dir,
                                  filename="aipipe.pt")
        if self.training_mode != "scratch" and os.path.realpath(os.path.dirname(source_path)) == os.path.realpath(self.output_dir):
            raise ValueError("Target output must not overwrite the shared pretrained checkpoint")
        source_provenance = {}
        if self.training_mode != "scratch":
            source_provenance = load_pretrained(agent, source_path, agent.checkpoint_metadata)
        provenance = {"mode": self.training_mode, "target_task": self.data_name,
                      "source_checkpoint": source_path if self.training_mode != "scratch" else None,
                      "source_provenance": source_provenance,
                      "target_seen_in_pretraining": self.data_name in source_provenance.get("training_tasks", [])}
        agent_training = {"frames": 0, "episodes": 0}
        training_started = time.monotonic()
        if self.training_mode != "pretrained":
            agent_training = trainer.train(aipipe_path, training_mode=self.training_mode)
        training_seconds = time.monotonic() - training_started if self.training_mode != "pretrained" else 0.0
        freeze_agent(agent)
        agent.save(aipipe_path, provenance=provenance)
        ai, trace = env.rollout(trainer.agent, epsilon=0.0)
        candidates = combine_pipelines(
            hi, ai, ctx, random.Random(self.seed), max_len=self.max_pipeline_len,
        )
        al_model = ScoreNN(env.data_dim, len(env.primitive_vocabulary)).to(self.device)
        if self.training_mode != "scratch":
            from .pretrain import load_alternative_evaluator
            load_alternative_evaluator(al_model,
                os.path.join(os.path.dirname(source_path), "alternative_evaluator.pt"),
                env.primitive_vocabulary, agent.checkpoint_metadata)
        selector = ActiveCandidateSelector(
            al_model, k_total=self.al_k, iterations=self.al_iterations, seed=self.seed,
        )
        adaptation_started = time.monotonic()
        selected = select_best_candidate(
            evaluator, candidates, selector=selector,
            dataset_feature=env.data_feature, vocabulary=env.primitive_vocabulary,
        )
        scorer_adaptation_seconds = time.monotonic() - adaptation_started
        al_path = os.path.join(self.output_dir, "alternative_evaluator.pt")
        save_alternative_evaluator(
            al_model, al_path, data_dim=env.data_dim,
            primitive_vocabulary=env.primitive_vocabulary,
            checkpoint_metadata=agent.checkpoint_metadata,
            provenance={
                "training_mode": self.training_mode,
                "source_checkpoint": provenance["source_checkpoint"],
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
            "training_mode": self.training_mode,
            "training_provenance": provenance,
            "training_seconds": training_seconds,
            "scorer_adaptation_seconds": scorer_adaptation_seconds,
            "agent_training": agent_training,
            "checkpoints": {"aipipe": aipipe_path, "alternative_evaluator": al_path},
        }
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(log, handle, indent=2)
        return log


__all__ = ["HAIPipe"]
