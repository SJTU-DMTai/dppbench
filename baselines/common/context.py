"""Shared dataset-context inference helpers for baseline pipeline builders."""
from __future__ import annotations

from baselines.common.pipeline import DataContext


# -----------------------------------------------------------------------------
# Context inference helpers
# -----------------------------------------------------------------------------
def _infer_rec_context(data_name: str, summary: dict, data) -> DataContext:
    interaction_df = data.interaction_df
    col_types = data.col_types or {}

    numeric_cols = [c for c in interaction_df.columns if col_types.get(c) == "numeric"]
    categorical_cols = [c for c in interaction_df.columns if col_types.get(c) == "categorical"]
    list_cols = [c for c in interaction_df.columns
                 if col_types.get(c) in ("numeric_list", "categorical_list")]
    text_cols = [c for c in interaction_df.columns if col_types.get(c) == "text"]

    # Add columns from user_df / item_df schemas as well so JoinTable downstream
    # operators can reference them by name.
    for side_df in (data.user_df, data.item_df):
        if side_df is None:
            continue
        for c in side_df.columns:
            if c in (data._user_id_col, data._item_id_col):
                continue
            t = col_types.get(c)
            if t == "numeric":
                numeric_cols.append(c)
            elif t == "categorical":
                categorical_cols.append(c)
            elif t in ("numeric_list", "categorical_list"):
                list_cols.append(c)
            elif t == "text":
                text_cols.append(c)

    # target column: prefer the task/model contract over name heuristics.
    feat_cfg = getattr(data, "model_cfg", {}).get("feature", {}) or {}
    configured_target = feat_cfg.get("target_col")
    target_col = configured_target if configured_target in interaction_df.columns else None
    for cand in ("rating", "stars", "label", "click", "is_click"):
        if target_col is not None:
            break
        if cand in interaction_df.columns:
            target_col = cand
            break
    # time column
    time_col = None
    for cand in ("timestamp", "time", "ts"):
        if cand in interaction_df.columns:
            time_col = cand
            break

    return DataContext(
        task_type="rec",
        data_name=data_name,
        numeric_cols=sorted(set(numeric_cols)),
        categorical_cols=sorted(set(categorical_cols)),
        list_cols=sorted(set(list_cols)),
        text_cols=sorted(set(text_cols)),
        target_col=target_col,
        id_col=None,
        time_col=time_col,
        user_col=data._user_id_col,
        item_col=data._item_id_col,
        has_user_df=data.user_df is not None,
        has_item_df=data.item_df is not None,
        aux_dfs=[],
    )


def _infer_tabular_context(data_name: str, summary: dict, data) -> DataContext:
    train_df = data.train_df
    target_col = data.target_col
    id_col = data.id_col

    num_cols, cat_cols = [], []
    for c in train_df.columns:
        if c in (target_col, id_col):
            continue
        if train_df[c].dtype.kind in ("i", "u", "f"):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    # Detect a temporal column heuristically (used by ExtractDateTimeFeature)
    time_col = None
    for cand in ("TransactionDT", "timestamp", "time"):
        if cand in train_df.columns:
            time_col = cand
            break

    aux_names = list((data.auxiliary_dfs or {}).keys())
    aux_names = [a for a in aux_names if data.auxiliary_dfs.get(a) is not None]

    # Sentinel rules used by CustomClean (heuristic per dataset)
    sentinel_rules: list[dict] = []
    if data_name == "home_credit":
        if "DAYS_EMPLOYED" in train_df.columns:
            sentinel_rules.append({"col": "DAYS_EMPLOYED", "value": 365243})
        if "CODE_GENDER" in train_df.columns:
            sentinel_rules.append({"col": "CODE_GENDER", "value": "XNA"})
        if "ORGANIZATION_TYPE" in train_df.columns:
            sentinel_rules.append({"col": "ORGANIZATION_TYPE", "value": "XNA"})

    return DataContext(
        task_type="tabular",
        data_name=data_name,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        list_cols=[],
        text_cols=[],
        target_col=target_col,
        id_col=id_col,
        time_col=time_col,
        aux_dfs=aux_names,
        sentinel_rules=sentinel_rules,
    )


def _infer_graph_context(data_name: str, summary: dict, data) -> DataContext:
    train_df = data.train_df
    target_col = getattr(data, "target_col", None)
    id_col = getattr(data, "id_col", None)

    numeric_cols, categorical_cols = [], []
    for c in train_df.columns:
        if c in (target_col, id_col):
            continue
        if train_df[c].dtype.kind in ("i", "u", "f"):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    time_col = None
    for cand in ("time_step", "timestamp", "time"):
        if cand in train_df.columns:
            time_col = cand
            break

    aux_names = list((data.auxiliary_dfs or {}).keys())
    aux_names = [a for a in aux_names if data.auxiliary_dfs.get(a) is not None]
    return DataContext(
        task_type="graph",
        data_name=data_name,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        list_cols=[],
        text_cols=[],
        target_col=target_col,
        id_col=id_col,
        time_col=time_col,
        aux_dfs=aux_names,
    )


