"""LearnedPruner -- AlphaClean §6.3.

Trains a Logistic Regression that maps featurised repairs to "is this repair
likely to appear in an optimal pipeline?" probabilities. The decision
threshold is swept down until the false-negative rate on the training set is
zero, which biases the model toward false positives (it would rather keep a
useless ca than throw away a useful one).

Falls back gracefully when sklearn is unavailable: in that case the pruner is
a permanent no-op (everything passes).
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - environment dependent
    LogisticRegression = None  # type: ignore[assignment]
    _HAS_SKLEARN = False

from .repair import Repair


@dataclass
class LearnedPruner:
    op_index: dict
    ctx: object  # DataContext (kept loose to avoid circular import)
    min_samples: int = 20
    refit_every: int = 16
    enabled: bool = True

    _features: List[np.ndarray] = field(default_factory=list)
    _labels: List[int] = field(default_factory=list)
    _model: Optional[object] = None
    _threshold: float = 0.0
    _fit_calls: int = 0
    _samples_since_last_fit: int = 0

    # ------------------------------------------------------------------
    def add_example(self, repair: Repair, label: int) -> None:
        if not self.enabled:
            return
        feat = repair.featurize(self.op_index, self.ctx)
        self._features.append(feat)
        self._labels.append(int(label))
        self._samples_since_last_fit += 1

    def add_pipeline_examples(self, best_ops: set, candidate_repairs: List[Repair]) -> None:
        """Bulk helper: positive iff op_name appears in current best ops."""
        for r in candidate_repairs:
            self.add_example(r, 1 if r.op_name in best_ops else 0)

    # ------------------------------------------------------------------
    def fit_if_ready(self) -> bool:
        if not self.enabled or not _HAS_SKLEARN:
            return False
        if len(self._features) < self.min_samples:
            return False
        if self._samples_since_last_fit < self.refit_every and self._model is not None:
            return False

        X = np.stack(self._features, axis=0)
        y = np.array(self._labels, dtype=np.int64)
        if len(np.unique(y)) < 2:
            return False  # need both classes

        try:
            model = LogisticRegression(max_iter=200, class_weight="balanced")
            model.fit(X, y)
        except Exception:
            return False

        # Threshold sweep: lower threshold until 0 false negatives on train.
        probs = model.predict_proba(X)[:, 1]
        thresholds = sorted(set(np.linspace(0.0, 1.0, 21).tolist()), reverse=True)
        chosen = 0.0
        for t in thresholds:
            preds = (probs >= t).astype(int)
            fn = int(((preds == 0) & (y == 1)).sum())
            if fn == 0:
                chosen = float(t)
                break

        self._model = model
        self._threshold = chosen
        self._fit_calls += 1
        self._samples_since_last_fit = 0
        return True

    # ------------------------------------------------------------------
    def predict(self, repair: Repair) -> bool:
        if not self.enabled or self._model is None:
            return True
        try:
            feat = repair.featurize(self.op_index, self.ctx).reshape(1, -1)
            prob = float(self._model.predict_proba(feat)[0, 1])
            return prob >= self._threshold
        except Exception:
            return True

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "model": self._model,
                    "threshold": self._threshold,
                    "fit_calls": self._fit_calls,
                    "n_samples": len(self._features),
                    "n_positive": int(sum(self._labels)),
                }, f)
        except Exception:
            pass

    @property
    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "n_samples": len(self._features),
            "n_positive": int(sum(self._labels)) if self._labels else 0,
            "fit_calls": self._fit_calls,
            "threshold": float(self._threshold),
            "model_ready": self._model is not None,
        }


__all__ = ["LearnedPruner"]
