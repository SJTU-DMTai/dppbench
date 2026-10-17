#!/usr/bin/env python
"""Pretrain pipeline agents on an explicit source-task set (validation rewards only)."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from baselines.common.training_utils import contract, resolved_model, isolate_training_data
from baselines.common.search_evaluator import SearchPipelineEvaluator
from baselines.common.utils import get_data_ctx


def names(text):
    return list(dict.fromkeys(n.strip() for n in text.split(",") if n.strip()))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", choices=("CtxPipe", "HAIPipe", "FlowPipe"), required=True)
    p.add_argument("--data_names", required=True, help="Comma-separated source tasks; may include later seen-task targets")
    p.add_argument("--held_out_data_names", default="", help="Optional disjoint unseen targets, forbidden as sources")
    p.add_argument("--model", required=True, help="One downstream model per shared agent")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--pretrained_dir", default="checkpoints/pretrained")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--small_n", type=int, default=1500)
    p.add_argument("--max_pipeline_len", type=int, default=10)
    p.add_argument("--frames", type=int, default=None, help="Total source interaction steps, not per dataset")
    p.add_argument("--epochs", type=int, default=20, help="FlowPipe source epochs")
    p.add_argument("--trajectories_per_combo", type=int, default=8)
    p.add_argument("--updates_per_combo", type=int, default=1)
    p.add_argument("--llm_model_path", default="/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B")
    p.add_argument("--context_cache_dir", default=None)
    p.add_argument("--al_candidates_per_task", type=int, default=40)
    p.add_argument("--fast_train", action="store_true", help="Use shared evaluator fast training overrides for source rewards")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def build_runner(args, task):
    if args.baseline == "CtxPipe":
        from baselines.CtxPipe.ctxpipe import CtxPipe as Runner
    elif args.baseline == "HAIPipe":
        from baselines.HAIPipe.haipipe import HAIPipe as Runner
    else:
        from baselines.FlowPipe.flowpipe import FlowPipe as Runner
    kwargs = dict(task_dir=str(ROOT / "dppbench/tasks" / task), data_name=task,
                  data_dir=str(Path(args.data_dir) / task / "data"), model_name=args.model,
                  device=args.device, seed=args.seed, training_mode="scratch", eval_full=False,
                  small_n=args.small_n, fast_train=args.fast_train,
                  output_dir=str(args.run_dir / "source_logs" / task))
    if args.baseline == "CtxPipe":
        kwargs.update(gte_model_path=args.llm_model_path, context_cache_dir=args.context_cache_dir)
    elif args.baseline == "HAIPipe":
        kwargs["max_pipeline_len"] = args.max_pipeline_len
    else:
        kwargs.update(llama_model_path=args.llm_model_path, context_cache_dir=args.context_cache_dir,
                      max_pipeline_len=args.max_pipeline_len)
    return Runner(**kwargs)


def build_sources(args):
    runners, evaluators, envs, contexts = [], [], [], []
    for task in args.sources:
        runner = build_runner(args, task)
        evaluator = SearchPipelineEvaluator(
            task_dir=runner.task_dir, data_name=task, data_dir=runner.data_dir,
            model_name=args.model, device=args.device, seed=args.seed,
            small_n=args.small_n, verbose=True,
        )
        evaluator.set_evaluate_std_test(False)
        isolate_training_data(evaluator)
        evaluator.set_fast_mode(args.fast_train)
        if resolved_model(evaluator) != args.model:
            raise ValueError(f"{task} does not resolve downstream model {args.model}")
        data = evaluator.load_data()
        ctx = get_data_ctx(task, data, evaluator.task_type)
        if args.baseline == "CtxPipe":
            from baselines.CtxPipe.env import PipelineEnv
            env = PipelineEnv(evaluator, ctx, max_steps=args.max_pipeline_len,
                              seed=args.seed, device=args.device,
                              gte_model_path=args.llm_model_path, context_cache_dir=args.context_cache_dir)
            # Share one frozen 8B encoder, not one resident copy per source dataset.
            if envs:
                env.embedder = envs[0].embedder
        elif args.baseline == "HAIPipe":
            from baselines.HAIPipe.env import HAIPipeEnv
            env = HAIPipeEnv(evaluator, ctx, model_name=args.model, seed=args.seed)
        else:
            from baselines.FlowPipe.env import FlowPipeEnv
            from baselines.FlowPipe.context import SemanticContextLoader
            if evaluator.task_type == "graph":
                raise ValueError("FlowPipe graph tasks are unsupported; remove graph sources")
            env = FlowPipeEnv(evaluator, ctx, max_steps=runner.max_steps, seed=args.seed)
            runner.precompute_context(device=args.device)
            contexts.append(SemanticContextLoader(args.context_cache_dir, runner.llm_name,
                             runner.ctx_dim, runner.llama_model_path).load(task))
        runners.append(runner)
        evaluators.append(evaluator)
        envs.append(env)
    return runners, evaluators, envs, contexts


def train(args):
    runners, evaluators, envs, contexts = build_sources(args)
    runner, env = runners[0], envs[0]
    encoder = None
    if args.baseline == "CtxPipe":
        encoder = {"path": os.path.realpath(args.llm_model_path), "dim": 4096}
    elif args.baseline == "FlowPipe":
        encoder = {"path": os.path.realpath(args.llm_model_path), "name": runner.llm_name, "dim": runner.ctx_dim}
    expected = contract(args.baseline, env, args.model, encoder)
    if any(contract(args.baseline, e, args.model, encoder) != expected for e in envs):
        raise ValueError("Source environments have incompatible agent contracts")
    provenance = {"training_tasks": args.sources, "held_out_tasks": args.heldout,
                  "model": args.model, "seed": args.seed, "mode": "cross_task_pretrain",
                  "small_n": args.small_n, "fast_train": args.fast_train}
    if args.baseline == "CtxPipe":
        from baselines.CtxPipe.agent import DQNAgent
        from baselines.CtxPipe.trainer import CtxPipeTrainer
        agent = DQNAgent(env.component_action_dims, env.primitive_vocab_size, env.n_logical_pipelines,
                         sequence_length=env.n_components, device=args.device, seed=args.seed,
                         lr=runner.lr, buffer_capacity=runner.buffer_capacity)
        agent.checkpoint_metadata, agent.provenance = expected, provenance
        trainer = CtxPipeTrainer(envs, agent, frames=args.frames or runner.frames,
            eps_start=runner.eps_start, eps_end=runner.eps_end, eps_decay=runner.eps_decay,
            batch_size=runner.batch_size, logical_batch_size=runner.logical_batch_size,
            backpropagate_interval=runner.backpropagate_interval,
            checkpoint_interval=runner.checkpoint_interval, output_dir=str(args.run_dir))
        summary = trainer.train()["training_stats"]
        agent.save(str(args.run_dir / "checkpoint.pt"))
    elif args.baseline == "FlowPipe":
        from baselines.FlowPipe.agent import GFNAgent
        from baselines.FlowPipe.trainer import FlowPipeTrainer
        agent = GFNAgent(env.state_dim, env.action_dim, hidden_dim=runner.hidden_dim,
            ctx_dim=runner.ctx_dim, learning_rate=runner.learning_rate, buffer_capacity=runner.buffer_capacity,
            failure_weight=runner.failure_weight, reward_temperature=runner.reward_temperature, device=args.device)
        agent.checkpoint_metadata, agent.provenance = expected, provenance
        trainer = FlowPipeTrainer(agent, list(zip(envs, contexts)), output_dir=str(args.run_dir),
            trajectories_per_combo=args.trajectories_per_combo, updates_per_combo=args.updates_per_combo,
            trajectory_batch_size=runner.trajectory_batch_size, seed=args.seed)
        trainer.train(args.epochs)
        agent.save(str(args.run_dir / "checkpoint.pt"))
        summary = {"optimizer_updates": trainer.optimizer_updates, "history": trainer.history}
    else:
        from baselines.HAIPipe.pretrain import CrossTaskPretrainer, save_alternative_evaluator
        from baselines.HAIPipe.active_learning import ScoreNN, pipeline_features, pretrain_score_model
        from baselines.HAIPipe.hi_pipeline import build_hi_pipeline
        from baselines.HAIPipe.combiner import combine_pipelines
        trainer = CrossTaskPretrainer(envs, frames=args.frames or runner.frames,
            learning_rate=runner.learning_rate, buffer_capacity=runner.buffer_capacity,
            batch_size=runner.batch_size, gamma=runner.gamma, train_interval=runner.train_interval,
            logic_batch_size=runner.logic_batch_size, epsilon_start=runner.epsilon_start,
            epsilon_end=runner.epsilon_end, epsilon_decay=runner.epsilon_decay,
            seed=args.seed, device=args.device)
        trainer.agent.checkpoint_metadata = expected
        summary = trainer.train(str(args.run_dir / "aipipe.pt"), held_out_tasks=args.heldout)
        ds, ps, ys = [], [], []
        rng = random.Random(args.seed)
        for e, r in zip(envs, runners):
            hi = build_hi_pipeline(e.ctx, rng, task_dir=r.task_dir, max_len=args.max_pipeline_len)
            ai, _ = e.rollout(trainer.agent)
            candidates = combine_pipelines(hi, ai, e.ctx, rng, max_len=args.max_pipeline_len)
            if len(candidates) > args.al_candidates_per_task:
                candidates = rng.sample(candidates, args.al_candidates_per_task)
            successful = []
            for candidate in candidates:
                ev = e.evaluator.evaluate(candidate)
                if ev.success and math.isfinite(ev.fitness):
                    successful.append((candidate, ev.fitness))
            if not successful:
                continue
            mean = np.mean([v for _, v in successful])
            scale = max(1.0, max(abs(v - mean) for _, v in successful))
            for candidate, score in successful:
                ds.append(e.data_feature)
                ps.append(pipeline_features(candidate, e.primitive_vocabulary))
                ys.append((score - mean) / scale)
        if len(ys) < 2:
            raise RuntimeError("Need at least two finite source HAI candidates to pretrain scorer")
        model = ScoreNN(env.data_dim, len(env.primitive_vocabulary)).to(args.device)
        losses = pretrain_score_model(model, np.stack(ds), np.stack(ps), np.asarray(ys))
        save_alternative_evaluator(model, str(args.run_dir / "alternative_evaluator.pt"),
            data_dim=env.data_dim, primitive_vocabulary=env.primitive_vocabulary,
            provenance=provenance, checkpoint_metadata=expected)
        trainer.agent.save(str(args.run_dir / "aipipe.pt"), provenance=provenance)
        summary.update(scorer_samples=len(ys), scorer_loss=losses[-1])
    source_evaluations = {e.data_name: e.n_unique_evaluations for e in evaluators}
    for evaluator in evaluators:
        evaluator._executor.cleanup()
    return {"provenance": provenance, "checkpoint_metadata": expected, "training": summary,
            "source_unique_evaluations": source_evaluations}


def main():
    args = parse_args()
    args.sources, args.heldout = names(args.data_names), names(args.held_out_data_names)
    if not args.sources or set(args.sources) & set(args.heldout):
        raise ValueError("Need nonempty source tasks disjoint from explicitly held-out tasks")
    if args.small_n < 0 or (args.frames is not None and args.frames <= 0) or args.epochs <= 0:
        raise ValueError("Training budgets must be positive (small_n=0 means full data)")
    args.device = f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu"
    args.run_dir = Path(args.pretrained_dir).resolve() / args.baseline / args.model
    if not args.overwrite and any(args.run_dir.glob("*.pt")):
        raise FileExistsError(f"Existing weights at {args.run_dir}; select another pretrained_dir or --overwrite")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.context_cache_dir = str(Path(args.context_cache_dir or args.run_dir / "context_cache").resolve())
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    start = time.monotonic()
    result = train(args)
    result["pretrain_seconds"] = time.monotonic() - start
    (args.run_dir / "pretrain.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
