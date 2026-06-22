"""Pipeline evaluator for CtxPipe.

Wraps SAGA's :class:`PipelineEvaluator` and adds a ``small_n`` parameter that
subsamples the data after each ``data.load_data()`` call so the RL training
loop can iterate cheaply over many candidate pipelines.

When ``small_n`` is ``None`` or ``0`` the evaluator behaves identically to
SAGA's evaluator (full data, full training).
"""
from __future__ import annotations

from typing import Optional

from baselines.SAGA.evaluator import EvaluationResult, PipelineEvaluator


class CtxPipeEvaluator(PipelineEvaluator):
    """PipelineEvaluator with optional small-data subsampling."""

    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir=None,
        metric_key: str = "auc",
        verbose: bool = False,
        small_n: Optional[int] = None,
        seed: int = 42,
        device: str = "cpu",
        model_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            task_dir=task_dir,
            data_name=data_name,
            data_dir=data_dir,
            metric_key=metric_key,
            verbose=verbose,
            device=device,
            model_name=model_name,
        )
        self.small_n = int(small_n) if small_n else 0
        self.seed = int(seed)

        # Patch the executor so each `data.load_data()` call subsamples.
        self._orig_make_data_instance = self._executor._make_data_instance
        if self.small_n > 0:
            self._executor._make_data_instance = self._patched_make_data_instance

    # ------------------------------------------------------------------
    def set_small_n(self, small_n: Optional[int]) -> None:
        """Toggle subsampling at runtime."""
        new_n = int(small_n) if small_n else 0
        self.small_n = new_n
        if new_n > 0:
            self._executor._make_data_instance = self._patched_make_data_instance
        else:
            self._executor._make_data_instance = self._orig_make_data_instance
        # Cache must be invalidated because identical YAMLs may now refer to a
        # different dataset size.
        self._cache.clear()
        self._executor._data = None

    # ------------------------------------------------------------------
    def _patched_make_data_instance(self):
        """Return a data instance whose ``load_data()`` subsamples in-place."""
        small_n = self.small_n
        seed = self.seed
        data = self._orig_make_data_instance()
        original_load_data = data.load_data
        original_load_std_test = getattr(data, "load_std_test_frozen", None)

        def patched_load_data(*args, **kwargs):
            ret = original_load_data(*args, **kwargs)
            try:
                _subsample_data_inplace(data, small_n=small_n, seed=seed)
            except Exception:
                # Subsampling is best-effort: never break training over it.
                pass
            return ret

        data.load_data = patched_load_data  # type: ignore[assignment]
        if callable(original_load_std_test):
            def patched_load_std_test(*args, **kwargs):
                ret = original_load_std_test(*args, **kwargs)
                try:
                    _subsample_data_inplace(data, small_n=small_n, seed=seed)
                except Exception:
                    # Subsampling is best-effort: never break training over it.
                    pass
                return ret

            data.load_std_test_frozen = patched_load_std_test  # type: ignore[assignment]
        return data


# ---------------------------------------------------------------------------
# Subsampling helpers
# ---------------------------------------------------------------------------
def _subsample_data_inplace(data, small_n: int, seed: int) -> None:
    """Truncate the in-memory DataFrames on ``data`` to ``small_n`` rows.

    Works for both ``TabularData`` (``train_df``/``test_df``/``auxiliary_dfs``)
    and ``RecData`` (``interaction_df``). User/item side tables are kept intact
    because they encode lookup information that the pipeline may rely on.
    """
    if small_n <= 0:
        return

    # ---- Tabular ----
    train_df = getattr(data, "train_df", None)
    if train_df is not None:
        if "__split__" in train_df.columns:
            data.train_df = _subsample_split_frame(
                train_df,
                small_n=small_n,
                seed=seed,
                target_col=getattr(data, "target_col", None),
            )
        elif len(train_df) > small_n:
            data.train_df = train_df.sample(
                n=small_n, random_state=seed
            ).reset_index(drop=True)
        test_df = getattr(data, "test_df", None)
        if test_df is not None:
            n_test = max(1, small_n // 2)
            if len(test_df) > n_test:
                data.test_df = test_df.sample(
                    n=n_test, random_state=seed
                ).reset_index(drop=True)

    # ---- Rec ----
    interaction_df = getattr(data, "interaction_df", None)
    if interaction_df is not None and len(interaction_df) > small_n:
        if "__split__" in interaction_df.columns:
            data.interaction_df = _subsample_rec_split_frame(
                interaction_df,
                data=data,
                small_n=small_n,
                seed=seed,
            )
        else:
            data.interaction_df = _subsample_rec_train_frame(
                interaction_df,
                small_n=small_n,
                seed=seed,
            )


def _subsample_split_frame(df, small_n: int, seed: int, target_col: str | None = None):
    """Sample train/std-test partitions separately while preserving both."""
    train = df[df["__split__"] != "std_test"]
    std = df[df["__split__"] == "std_test"]

    if len(train) > small_n:
        train = train.sample(n=small_n, random_state=seed)

    n_std = max(1, small_n // 2)
    if len(std) > n_std:
        if target_col and target_col in std.columns and std[target_col].nunique(dropna=True) > 1:
            pieces = []
            per_class = max(1, n_std // int(std[target_col].nunique(dropna=True)))
            for _, group in std.groupby(target_col, dropna=False):
                pieces.append(group.sample(n=min(len(group), per_class), random_state=seed))
            std = _sample_to_budget(pieces, n_std, seed)
        else:
            std = std.sample(n=n_std, random_state=seed)

    return _restore_order([train, std])


def _subsample_rec_train_frame(df, small_n: int, seed: int):
    """Keep a coherent recent slice for recommendation interactions."""
    time_col = next((c for c in ("timestamp", "time", "ts") if c in df.columns), None)
    if len(df) <= small_n:
        return df.reset_index(drop=True)
    if time_col is not None:
        return df.sort_values(time_col).tail(small_n).reset_index(drop=True)
    return df.sample(n=small_n, random_state=seed).reset_index(drop=True)


def _subsample_rec_split_frame(df, data, small_n: int, seed: int):
    train = df[df["__split__"] != "std_test"]
    std = df[df["__split__"] == "std_test"]

    train = _subsample_rec_train_frame(train, small_n=small_n, seed=seed)

    if len(std) > small_n:
        std = _subsample_rec_std_test_frame(std, data=data, small_n=small_n, seed=seed)

    sampled = _restore_order([train, std])
    _sync_std_test_negatives(data, sampled)
    return sampled


def _subsample_rec_std_test_frame(std, data, small_n: int, seed: int):
    feat_cfg = getattr(data, "model_cfg", {}).get("feature", {}) if getattr(data, "model_cfg", None) else {}
    target_col = feat_cfg.get("target_col")
    label_rule = feat_cfg.get("label_rule", {}) or {}
    positive_label = label_rule.get("positive_label", 1)
    user_col = getattr(data, "_user_id_col", "user_id")

    if target_col not in std.columns or user_col not in std.columns:
        return std.sample(n=small_n, random_state=seed).reset_index(drop=True)

    positives = std[std[target_col] == positive_label]
    if positives.empty:
        return std.sample(n=small_n, random_state=seed).reset_index(drop=True)

    # Frozen rec std-test is usually one positive plus about 100 negatives per
    # user. Sample by positive users first so every search-time test set keeps
    # valid positive/negative groups instead of becoming all-negative by chance.
    positives_per_budget = max(1, small_n // 101)
    n_pos = min(len(positives), positives_per_budget)
    chosen_pos = positives.sample(n=n_pos, random_state=seed)
    chosen_users = set(chosen_pos[user_col].dropna().unique())
    std_for_users = std[std[user_col].isin(chosen_users)]

    if len(std_for_users) <= small_n:
        return std_for_users.reset_index(drop=True)

    selected_pos = std_for_users[std_for_users[target_col] == positive_label]
    selected_neg = std_for_users[std_for_users[target_col] != positive_label]
    n_neg = max(0, small_n - len(selected_pos))
    if len(selected_neg) > n_neg:
        selected_neg = selected_neg.sample(n=n_neg, random_state=seed)
    return _restore_order([selected_pos, selected_neg])


def _sample_to_budget(frames, budget: int, seed: int):
    import pandas as pd

    df = pd.concat(frames, ignore_index=False)
    if len(df) > budget:
        df = df.sample(n=budget, random_state=seed)
    return df


def _restore_order(frames):
    import pandas as pd

    out = pd.concat([f for f in frames if f is not None and len(f) > 0], ignore_index=False)
    return out.sort_index().reset_index(drop=True)


def _sync_std_test_negatives(data, sampled):
    neg_df = getattr(data, "std_test_negatives_df", None)
    if neg_df is None or "__split__" not in sampled.columns:
        return
    feat_cfg = getattr(data, "model_cfg", {}).get("feature", {}) if getattr(data, "model_cfg", None) else {}
    target_col = feat_cfg.get("target_col")
    label_rule = feat_cfg.get("label_rule", {}) or {}
    negative_label = label_rule.get("negative_label", 0)
    if target_col and target_col in sampled.columns:
        negatives = sampled[
            (sampled["__split__"] == "std_test") & (sampled[target_col] == negative_label)
        ].drop(columns="__split__", errors="ignore")
        data.std_test_negatives_df = negatives.reset_index(drop=True)


__all__ = ["CtxPipeEvaluator", "EvaluationResult"]
