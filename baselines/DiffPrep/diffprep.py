"""Top-level DiffPrep orchestrator.

Wraps:

  * :class:`DiffPrepEvaluator` -- runs the real downstream model on a YAML
    pipeline (LightGBM for tabular, DIN for rec).
  * :class:`ContinuousPipeline` -- the differentiable pipeline parameters.
  * :class:`TabularSurrogate` / :class:`RecSurrogate` -- the inner-loop
    learner whose validation loss provides the signal for ``tau``.
  * :class:`DiffPrepTrainer` -- the bilevel optimisation loop.
  * :func:`discretize` -- argmax projection from continuous parameters to a
    legal :class:`Pipeline`.

Public entry: :class:`DiffPrep`.run() returns a dict whose schema is aligned
with :class:`baselines.CtxPipe.ctxpipe.CtxPipe`.run().
"""
from __future__ import annotations

import json
import math
import os
import random as _random
import time
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.config import (
    default_config_path,
    load_baseline_config,
    resolve_config_value,
)
from baselines.SAGA.pipeline import DataContext, Pipeline
from baselines.SAGA.pipeline_constraints import is_legal
from baselines.SAGA.saga import _infer_rec_context, _infer_tabular_context

from .discretizer import argmax_op_names, discretize, slot_order
from .evaluator import DiffPrepEvaluator
from .search_space import ContinuousPipeline
from .slot_planner import make_slots
from .surrogate import RecSurrogate, TabularSurrogate
from .trainer import DiffPrepTrainer


CONFIG_KEYS = (
    "n_epochs",
    "small_n",
    "lr_w",
    "lr_alpha",
    "eps_finite_diff",
    "flex",
    "eval_full",
    "max_features",
    "batch_size",
    "val_ratio",
    "continuous_init_scale",
    "second_order",
    "sgd_momentum",
    "surrogate_hidden_dim",
    "rec_emb_dim",
    "gumbel_tau",
    "hard_sample",
    "seed",
)


# ---------------------------------------------------------------------------
# Helpers: extract numeric feature tensors + labels from a loaded data object.
# ---------------------------------------------------------------------------
def _zscore(arr: np.ndarray) -> np.ndarray:
    """Column-wise z-score with safe std and clipping."""
    if arr.size == 0:
        return arr
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (arr - mu) / sd
def _tabular_feature_tensors(data, ctx: DataContext, max_features: int,
                             device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_df = data.train_df
    target = ctx.target_col
    if target is None or target not in train_df.columns:
        raise RuntimeError(
            "DiffPrep tabular surrogate needs a target column to train on; "
            f"none found for dataset {ctx.data_name}."
        )

    cols = [c for c in ctx.numeric_cols if c != target][:max_features]
    if not cols:
        # Synthesise a single zero column so the surrogate can still run.
        feat = np.zeros((len(train_df), 1), dtype=np.float32)
    else:
        feat = train_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    mask = np.isnan(feat).astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    feat = _zscore(feat)
    feat = np.clip(feat, -5.0, 5.0).astype(np.float32)

    y = pd.to_numeric(train_df[target], errors="coerce").to_numpy(dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0)
    # Binary-ize if labels are floats.
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        thresh = float(np.median(y))
        y = (y > thresh).astype(np.float32)

    x_t = torch.tensor(feat, dtype=torch.float32, device=device)
    m_t = torch.tensor(mask, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    return x_t, m_t, y_t


def _rec_feature_tensors(data, ctx: DataContext, max_features: int,
                         device: torch.device
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    inter = data.interaction_df
    user_col = ctx.user_col
    item_col = ctx.item_col
    target = ctx.target_col

    if target is None or target not in inter.columns:
        raise RuntimeError(
            f"DiffPrep rec surrogate needs a target column for {ctx.data_name}."
        )

    # Encode user/item ids with hash mod to keep table small.
    n_users = max(int(inter[user_col].astype("category").cat.codes.max() + 1), 2) \
        if user_col in inter.columns else 2
    n_items = max(int(inter[item_col].astype("category").cat.codes.max() + 1), 2) \
        if item_col in inter.columns else 2

    if user_col in inter.columns:
        u_codes = inter[user_col].astype("category").cat.codes.to_numpy(dtype=np.int64)
    else:
        u_codes = np.zeros(len(inter), dtype=np.int64)
    if item_col in inter.columns:
        i_codes = inter[item_col].astype("category").cat.codes.to_numpy(dtype=np.int64)
    else:
        i_codes = np.zeros(len(inter), dtype=np.int64)

    cols = [c for c in ctx.numeric_cols if c not in (target, user_col, item_col)][:max_features]
    if cols:
        feat = inter[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    else:
        feat = np.zeros((len(inter), 1), dtype=np.float32)
    mask = np.isnan(feat).astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    feat = _zscore(feat)
    feat = np.clip(feat, -5.0, 5.0).astype(np.float32)

    y = pd.to_numeric(inter[target], errors="coerce").to_numpy(dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0)
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        thresh = float(np.median(y))
        y = (y > thresh).astype(np.float32)

    return (
        torch.tensor(feat, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
        torch.tensor(u_codes, dtype=torch.long, device=device),
        torch.tensor(i_codes, dtype=torch.long, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
        n_users,
        n_items,
    )


def _train_val_split(n: int, val_ratio: float = 0.3, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(int(n * val_ratio), 1)
    return idx[n_val:], idx[:n_val]


# ---------------------------------------------------------------------------
# DiffPrep main class
# ---------------------------------------------------------------------------
class DiffPrep:
    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir: Optional[str] = None,
        n_epochs: Optional[int] = None,
        small_n: Optional[int] = None,
        lr_w: Optional[float] = None,
        lr_alpha: Optional[float] = None,
        eps_finite_diff: Optional[float] = None,
        flex: Optional[bool] = None,
        eval_full: Optional[bool] = None,
        max_features: Optional[int] = None,
        batch_size: Optional[int] = None,
        val_ratio: Optional[float] = None,
        continuous_init_scale: Optional[float] = None,
        second_order: Optional[bool] = None,
        sgd_momentum: Optional[float] = None,
        surrogate_hidden_dim: Optional[int] = None,
        rec_emb_dim: Optional[int] = None,
        gumbel_tau: Optional[float] = None,
        hard_sample: Optional[bool] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        output_dir: Optional[str] = None,
        device: str = "cpu",
        config_path: Optional[str] = None,
    ) -> None:
        cfg = load_baseline_config(
            config_path or default_config_path(__file__), CONFIG_KEYS
        )
        n_epochs = resolve_config_value(cfg, "n_epochs", n_epochs)
        small_n = resolve_config_value(cfg, "small_n", small_n)
        lr_w = resolve_config_value(cfg, "lr_w", lr_w)
        lr_alpha = resolve_config_value(cfg, "lr_alpha", lr_alpha)
        eps_finite_diff = resolve_config_value(
            cfg, "eps_finite_diff", eps_finite_diff
        )
        flex = resolve_config_value(cfg, "flex", flex)
        eval_full = resolve_config_value(cfg, "eval_full", eval_full)
        max_features = resolve_config_value(cfg, "max_features", max_features)
        batch_size = resolve_config_value(cfg, "batch_size", batch_size)
        val_ratio = resolve_config_value(cfg, "val_ratio", val_ratio)
        continuous_init_scale = resolve_config_value(
            cfg, "continuous_init_scale", continuous_init_scale
        )
        second_order = resolve_config_value(cfg, "second_order", second_order)
        sgd_momentum = resolve_config_value(cfg, "sgd_momentum", sgd_momentum)
        surrogate_hidden_dim = resolve_config_value(
            cfg, "surrogate_hidden_dim", surrogate_hidden_dim
        )
        rec_emb_dim = resolve_config_value(cfg, "rec_emb_dim", rec_emb_dim)
        gumbel_tau = resolve_config_value(cfg, "gumbel_tau", gumbel_tau)
        hard_sample = resolve_config_value(cfg, "hard_sample", hard_sample)
        seed = resolve_config_value(cfg, "seed", seed)

        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir
        self.n_epochs = int(n_epochs)
        self.small_n = int(small_n) if small_n else 0
        self.lr_w = float(lr_w)
        self.lr_alpha = float(lr_alpha)
        self.eps_finite_diff = float(eps_finite_diff)
        self.flex = bool(flex)
        self.eval_full = bool(eval_full)
        self.max_features = int(max_features)
        self.batch_size = int(batch_size)
        self.val_ratio = float(val_ratio)
        self.continuous_init_scale = float(continuous_init_scale)
        self.second_order = bool(second_order)
        self.sgd_momentum = float(sgd_momentum)
        self.surrogate_hidden_dim = int(surrogate_hidden_dim)
        self.rec_emb_dim = int(rec_emb_dim)
        self.gumbel_tau = float(gumbel_tau)
        self.hard_sample = bool(hard_sample)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "DiffPrep", data_name
        )
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device(device)  # honours --gpu_id; falls back to CPU
        self._device_str = device

    # ------------------------------------------------------------------
    def _build_context(self, evaluator: DiffPrepEvaluator) -> Tuple[DataContext, object]:
        data = evaluator._executor._load_data()
        if evaluator.task_type == "rec":
            ctx = _infer_rec_context(self.data_name, {}, data)
        else:
            ctx = _infer_tabular_context(self.data_name, {}, data)
        return ctx, data

    # ------------------------------------------------------------------
    def run(self) -> dict:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        _random.seed(self.seed)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        t0 = time.time()

        # 1) train-time evaluator with subsampling
        train_evaluator = DiffPrepEvaluator(
            task_dir=self.task_dir,
            data_name=self.data_name,
            data_dir=self.data_dir,
            verbose=self.verbose,
            small_n=self.small_n,
            seed=self.seed,
            device=self._device_str,
        )
        ctx, data = self._build_context(train_evaluator)
        task_type = ctx.task_type

        if self.verbose:
            print("=" * 60)
            print(f"[DiffPrep] dataset={self.data_name}  task={task_type}")
            print(f"[DiffPrep] numeric={len(ctx.numeric_cols)}  "
                  f"categorical={len(ctx.categorical_cols)}")
            print(f"[DiffPrep] target={ctx.target_col}  time={ctx.time_col}  "
                  f"id={ctx.id_col}")
            print(f"[DiffPrep] flex={self.flex}  small_n={self.small_n or 'OFF'}  "
                  f"n_epochs={self.n_epochs}")
            print("=" * 60)

        # 2) Build slots
        slots = make_slots(task_type, ctx)
        if self.verbose:
            print(f"[DiffPrep] {len(slots)} slots: "
                  + ", ".join(f"{s.category.value}({s.kind},m={s.n_candidates})" for s in slots))

        # 3) Build continuous pipeline + surrogate
        cont = ContinuousPipeline(
            slots, flex=self.flex, init_scale=self.continuous_init_scale
        ).to(self.device)

        if task_type == "rec":
            (feat, mask, u_ids, i_ids, y, n_users, n_items) = _rec_feature_tensors(
                data, ctx, self.max_features, self.device,
            )
            surrogate: nn.Module = RecSurrogate(
                n_users=n_users,
                n_items=n_items,
                feat_dim=feat.shape[1],
                emb_dim=self.rec_emb_dim,
                hidden=self.surrogate_hidden_dim,
            ).to(self.device)

            tr_idx, va_idx = _train_val_split(
                len(y), val_ratio=self.val_ratio, seed=self.seed
            )
            tr_idx_t = torch.tensor(tr_idx, dtype=torch.long, device=self.device)
            va_idx_t = torch.tensor(va_idx, dtype=torch.long, device=self.device)

            def make_step_fn(idx_t: torch.Tensor):
                def step(continuous: ContinuousPipeline, model: nn.Module):
                    n = idx_t.numel()
                    if n == 0:
                        zero = torch.zeros((), device=self.device, requires_grad=True)
                        return zero, 0.0
                    bsz = min(self.batch_size, n)
                    sel = idx_t[torch.randperm(n, device=self.device)[:bsz]]
                    fx = feat[sel]
                    mx = mask[sel]
                    ux = u_ids[sel]
                    ix = i_ids[sel]
                    yy = y[sel]
                    out = continuous(
                        fx, mx,
                        gumbel_tau=self.gumbel_tau,
                        hard_sample=self.hard_sample,
                    )
                    logits = model(ux, ix, out)
                    loss = F.binary_cross_entropy_with_logits(logits, yy)
                    return loss
                return step

            train_step = lambda c, m: make_step_fn(tr_idx_t)(c, m)

            def val_step(continuous: ContinuousPipeline, model: nn.Module):
                n = va_idx_t.numel()
                if n == 0:
                    z = torch.zeros((), device=self.device, requires_grad=True)
                    return z, 0.0
                bsz = min(self.batch_size, n)
                sel = va_idx_t[torch.randperm(n, device=self.device)[:bsz]]
                fx = feat[sel]
                mx = mask[sel]
                ux = u_ids[sel]
                ix = i_ids[sel]
                yy = y[sel]
                out = continuous(
                    fx, mx,
                    gumbel_tau=self.gumbel_tau,
                    hard_sample=self.hard_sample,
                )
                logits = model(ux, ix, out)
                loss = F.binary_cross_entropy_with_logits(logits, yy)
                with torch.no_grad():
                    pred = (torch.sigmoid(logits) > 0.5).float()
                    acc = (pred == yy).float().mean().item()
                return loss, acc
        else:
            feat, mask, y = _tabular_feature_tensors(
                data, ctx, self.max_features, self.device,
            )
            surrogate = TabularSurrogate(
                in_dim=feat.shape[1], hidden=self.surrogate_hidden_dim
            ).to(self.device)
            tr_idx, va_idx = _train_val_split(
                len(y), val_ratio=self.val_ratio, seed=self.seed
            )
            tr_idx_t = torch.tensor(tr_idx, dtype=torch.long, device=self.device)
            va_idx_t = torch.tensor(va_idx, dtype=torch.long, device=self.device)

            def train_step(continuous: ContinuousPipeline, model: nn.Module):
                n = tr_idx_t.numel()
                if n == 0:
                    return torch.zeros((), device=self.device, requires_grad=True)
                bsz = min(self.batch_size, n)
                sel = tr_idx_t[torch.randperm(n, device=self.device)[:bsz]]
                fx = feat[sel]
                mx = mask[sel]
                yy = y[sel]
                out = continuous(
                    fx, mx,
                    gumbel_tau=self.gumbel_tau,
                    hard_sample=self.hard_sample,
                )
                logits = model(out)
                loss = F.binary_cross_entropy_with_logits(logits, yy)
                return loss

            def val_step(continuous: ContinuousPipeline, model: nn.Module):
                n = va_idx_t.numel()
                if n == 0:
                    z = torch.zeros((), device=self.device, requires_grad=True)
                    return z, 0.0
                bsz = min(self.batch_size, n)
                sel = va_idx_t[torch.randperm(n, device=self.device)[:bsz]]
                fx = feat[sel]
                mx = mask[sel]
                yy = y[sel]
                out = continuous(
                    fx, mx,
                    gumbel_tau=self.gumbel_tau,
                    hard_sample=self.hard_sample,
                )
                logits = model(out)
                loss = F.binary_cross_entropy_with_logits(logits, yy)
                with torch.no_grad():
                    pred = (torch.sigmoid(logits) > 0.5).float()
                    acc = (pred == yy).float().mean().item()
                return loss, acc

        # 4) Bilevel training
        trainer = DiffPrepTrainer(
            continuous=cont,
            surrogate=surrogate,
            train_step_fn=train_step,
            val_step_fn=val_step,
            argmax_op_fn=argmax_op_names,
            n_epochs=self.n_epochs,
            lr_w=self.lr_w,
            lr_alpha=self.lr_alpha,
            eps_finite_diff=self.eps_finite_diff,
            second_order=self.second_order,
            sgd_momentum=self.sgd_momentum,
            verbose=self.verbose,
        )
        history = trainer.train()

        # 5) Discretise
        rng = _random.Random(self.seed)
        pipe = discretize(cont, slots, ctx, rng=rng)
        discrete_slot_order = slot_order(cont)

        # 6) Persist artefacts
        out_yaml_path = os.path.join(self.output_dir, "best_pipeline.yaml")
        with open(out_yaml_path, "w", encoding="utf-8") as f:
            f.write(pipe.to_yaml())

        weights_path = os.path.join(self.output_dir, "pipeline_weights.pt")
        try:
            torch.save({
                "tau": cont.tau.detach().cpu(),
                "theta": cont.theta.detach().cpu() if cont.theta is not None else None,
                "alpha": cont.alpha().detach().cpu(),
                "discrete_slot_order": discrete_slot_order,
                "surrogate": surrogate.state_dict(),
            }, weights_path)
        except Exception as e:
            if self.verbose:
                print(f"[DiffPrep] warning: failed to save pipeline weights: {e}")
            weights_path = None

        history_path = os.path.join(self.output_dir, "search_history.json")
        try:
            trainer.save_history(history_path)
        except Exception as e:
            if self.verbose:
                print(f"[DiffPrep] warning: failed to save search history: {e}")

        # 7) Final evaluation on real downstream model
        best_fitness: Optional[float] = None
        best_metrics: dict = {}
        eval_error: Optional[str] = None
        if self.eval_full:
            try:
                full_eval = DiffPrepEvaluator(
                    task_dir=self.task_dir,
                    data_name=self.data_name,
                    data_dir=self.data_dir,
                    verbose=self.verbose,
                    small_n=0,
                    seed=self.seed,
                    device=self._device_str,
                )
                ev = full_eval.evaluate(pipe)
                if ev.success:
                    best_fitness = float(ev.fitness)
                    best_metrics = dict(ev.metrics or {})
                else:
                    eval_error = ev.error
                    if self.verbose:
                        print(f"[DiffPrep] full eval failed: {ev.error}")
            except Exception as e:
                eval_error = f"{type(e).__name__}: {e}"
                if self.verbose:
                    print(f"[DiffPrep] full eval raised: {e}")

        duration = time.time() - t0

        if self.verbose:
            print("=" * 60)
            print(f"[DiffPrep] DONE in {duration:.1f}s")
            fit_str = f"{best_fitness:.4f}" if isinstance(best_fitness, float) else "n/a"
            print(f"[DiffPrep] best fitness = {fit_str}")
            print(f"[DiffPrep] best metrics = {best_metrics}")
            print(f"[DiffPrep] best pipeline saved to: {out_yaml_path}")
            print(f"[DiffPrep] weights saved to: {weights_path}")
            print(f"[DiffPrep] history saved to: {history_path}")
            print(f"[DiffPrep] unique evaluations: "
                  f"{train_evaluator.n_unique_evaluations}")
            print("=" * 60)

        return {
            "best_pipeline_yaml": pipe.to_yaml(),
            "best_pipeline_path": out_yaml_path,
            "best_fitness": best_fitness,
            "best_metrics": best_metrics,
            "eval_error": eval_error,
            "is_legal": is_legal(pipe, ctx.task_type),
            "final_pipeline_ops": [s.op for s in pipe.steps],
            "flex_enabled": bool(cont.flex),
            "alpha": cont.alpha().detach().cpu().tolist(),
            "discrete_slot_order": discrete_slot_order,
            "search_history": [
                {
                    "epoch": rec.epoch,
                    "train_loss": rec.train_loss,
                    "val_loss": rec.val_loss,
                    "val_acc": rec.val_acc,
                    "argmax_pipeline": rec.argmax_pipeline,
                    "duration_seconds": rec.duration_seconds,
                }
                for rec in history
            ],
            "n_unique_evaluations": train_evaluator.n_unique_evaluations,
            "duration_seconds": duration,
            "output_dir": self.output_dir,
            "weights_path": weights_path,
        }


__all__ = ["DiffPrep"]
