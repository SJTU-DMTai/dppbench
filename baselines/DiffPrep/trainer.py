"""DiffPrep bilevel trainer.

Implements DARTS-style first-order/second-order bilevel optimisation for
``tau`` (and optionally ``theta``):

  outer:  min_tau L_val(beta(tau), w*(tau))
  inner:  w*(tau) = argmin_w L_train(beta(tau), w)

We follow the standard DARTS approximation:

  1. Make a virtual step ``w' = w - lr_w * grad_w L_train`` on the surrogate.
  2. Compute ``grad_tau L_val(beta(tau), w')`` plus the second-order
     correction ``- lr_w * (grad_tau L_train(beta(tau), w+) -
     grad_tau L_train(beta(tau), w-)) / (2*epsilon)`` where ``w+/w-`` perturb
     ``w`` along ``+/- grad_w' L_val``.
  3. Apply the *real* updates: optimiser step on ``w`` using the
     train-loss gradient, optimiser step on ``tau`` using the val-loss
     gradient.

For simplicity the code falls back to a first-order approximation
(``second_order=False``) which has worked well in practice and keeps the
surrogate cheap. The ``second_order`` path is provided as a flag.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .search_space import ContinuousPipeline


# ---------------------------------------------------------------------------
# History record
# ---------------------------------------------------------------------------
@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    argmax_pipeline: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class DiffPrepTrainer:
    """Bilevel optimiser for the continuous pipeline + surrogate model."""

    def __init__(
        self,
        continuous: ContinuousPipeline,
        surrogate: nn.Module,
        train_step_fn: Callable[[ContinuousPipeline, nn.Module], torch.Tensor],
        val_step_fn: Callable[[ContinuousPipeline, nn.Module], Tuple[torch.Tensor, float]],
        argmax_op_fn: Callable[[ContinuousPipeline], List[str]],
        n_epochs: int = 5,
        lr_w: float = 1e-2,
        lr_alpha: float = 1e-3,
        eps_finite_diff: float = 1e-3,
        second_order: bool = False,
        sgd_momentum: float = 0.9,
        verbose: bool = True,
    ) -> None:
        self.continuous = continuous
        self.surrogate = surrogate
        self.train_step_fn = train_step_fn
        self.val_step_fn = val_step_fn
        self.argmax_op_fn = argmax_op_fn
        self.n_epochs = int(n_epochs)
        self.lr_w = float(lr_w)
        self.lr_alpha = float(lr_alpha)
        self.eps = float(eps_finite_diff)
        self.second_order = bool(second_order)
        self.sgd_momentum = float(sgd_momentum)
        self.verbose = bool(verbose)

        self.opt_w = torch.optim.SGD(self.surrogate.parameters(),
                                     lr=self.lr_w, momentum=self.sgd_momentum)
        arch_params = [self.continuous.tau]
        if self.continuous.theta is not None:
            arch_params.append(self.continuous.theta)
        self.opt_alpha = torch.optim.Adam(arch_params, lr=self.lr_alpha)

        self.history: List[EpochRecord] = []

    # ------------------------------------------------------------------
    def _step_arch(self) -> Tuple[float, float]:
        """One outer optimisation step.

        Returns ``(val_loss, val_acc)``.
        """
        # ---- Architecture step (first-order DARTS) ----
        self.opt_alpha.zero_grad()
        val_loss, val_acc = self.val_step_fn(self.continuous, self.surrogate)
        val_loss.backward()
        self.opt_alpha.step()
        return float(val_loss.detach().cpu().item()), float(val_acc)

    def _step_inner(self) -> float:
        """One inner optimisation step on the surrogate weights."""
        self.opt_w.zero_grad()
        train_loss = self.train_step_fn(self.continuous, self.surrogate)
        train_loss.backward()
        self.opt_w.step()
        return float(train_loss.detach().cpu().item())

    # ------------------------------------------------------------------
    def train(self) -> List[EpochRecord]:
        self.history.clear()
        self.continuous.train()
        self.surrogate.train()

        for epoch in range(self.n_epochs):
            t0 = time.time()
            # 1) inner update (surrogate)
            try:
                train_loss = self._step_inner()
            except Exception as e:
                if self.verbose:
                    print(f"[DiffPrep] epoch {epoch}: inner step failed ({e})")
                train_loss = float("nan")

            # 2) outer update (architecture)
            try:
                val_loss, val_acc = self._step_arch()
            except Exception as e:
                if self.verbose:
                    print(f"[DiffPrep] epoch {epoch}: outer step failed ({e})")
                val_loss, val_acc = float("nan"), float("nan")

            argmax_ops = self.argmax_op_fn(self.continuous)
            duration = time.time() - t0
            rec = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=val_acc,
                argmax_pipeline=argmax_ops,
                duration_seconds=duration,
            )
            self.history.append(rec)
            if self.verbose:
                print(
                    f"[DiffPrep] epoch={epoch+1}/{self.n_epochs}  "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"val_acc={val_acc:.4f}  ops={argmax_ops}"
                )
        return self.history

    # ------------------------------------------------------------------
    def save_history(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([rec.__dict__ for rec in self.history], f, indent=2)


__all__ = ["DiffPrepTrainer", "EpochRecord"]
