"""Shared dataset context objects and inference helpers for baselines."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace


@dataclass
class TableContext:
    """Schema metadata for one source table exposed to baselines."""

    name: str
    role: str                         # primary / secondary
    shape: tuple[int, int] | None = None
    dtypes: dict[str, str] = field(default_factory=dict)
    join_keys_to_primary: list[dict[str, str]] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return list(self.dtypes.keys())

    @property
    def dtypes_summary(self) -> dict[str, int]:
        return dict(Counter(self.dtypes.values()))

    @property
    def null_ratio(self) -> float:
        return 0.0

    @property
    def numeric_cols(self) -> list[str]:
        return [c for c, dtype in self.dtypes.items() if _is_numeric_dtype(dtype)]

    @property
    def categorical_cols(self) -> list[str]:
        return [c for c in self.columns if c not in set(self.numeric_cols)]

    @property
    def list_cols(self) -> list[str]:
        return []

    @property
    def text_cols(self) -> list[str]:
        return []

    @property
    def target_col(self) -> str | None:
        return _first_existing(
            self.columns,
            ("target", "label", "rating", "stars", "click", "is_click", "y"),
        )

    @property
    def id_col(self) -> str | None:
        for cand in ("id", "loan_id", "SK_ID_CURR", "user_id", "item_id", "node_id"):
            if cand in self.columns:
                return cand
        for col in self.columns:
            low = col.lower()
            if low.endswith("_id") or low.endswith("id"):
                return col
        return None

    @property
    def time_col(self) -> str | None:
        return _infer_time_col_from_columns(self.columns)


@dataclass
class DataContext:
    task_type: str
    data_name: str
    primary_table: str                 # YAML 中 s0 对应的主表表名
    secondary_tables: list[str] = field(default_factory=list)        # YAML 中 s1, s2, ... 对应的次表
    tables: dict[str, TableContext] = field(default_factory=dict)    # 每张 source table 的 schema/profile

    @property
    def primary_context(self) -> TableContext | None:
        return self.tables.get(self.primary_table)

    @property
    def aux_dfs(self) -> list[str]:
        return list(self.secondary_tables)

    @property
    def numeric_cols(self) -> list[str]:
        return _dedupe(
            col
            for table in self.tables.values()
            for col in table.numeric_cols
            if col not in _special_cols(self)
        )

    @property
    def categorical_cols(self) -> list[str]:
        return _dedupe(
            col
            for table in self.tables.values()
            for col in table.categorical_cols
            if col not in _special_cols(self)
        )

    @property
    def list_cols(self) -> list[str]:
        return []

    @property
    def text_cols(self) -> list[str]:
        return []

    @property
    def target_col(self) -> str | None:
        table = self.primary_context
        return table.target_col if table is not None else None

    @property
    def id_col(self) -> str | None:
        table = self.primary_context
        return table.id_col if table is not None else None

    @property
    def time_col(self) -> str | None:
        table = self.primary_context
        return table.time_col if table is not None else None

    @property
    def user_col(self) -> str:
        table = self.primary_context
        cols = table.columns if table is not None else []
        return _first_existing(cols, ("user_id", "user", "uid")) or "user_id"

    @property
    def item_col(self) -> str:
        table = self.primary_context
        cols = table.columns if table is not None else []
        return _first_existing(cols, ("item_id", "item", "iid", "movie_id")) or "item_id"

    @property
    def has_user_df(self) -> bool:
        return "user" in self.tables

    @property
    def has_item_df(self) -> bool:
        return "item" in self.tables

    @property
    def sentinel_rules(self) -> list[dict]:
        if self.data_name != "home_credit":
            return []
        table = self.primary_context
        cols = set(table.columns if table is not None else [])
        rules = []
        if "DAYS_EMPLOYED" in cols:
            rules.append({"col": "DAYS_EMPLOYED", "value": 365243})
        if "CODE_GENDER" in cols:
            rules.append({"col": "CODE_GENDER", "value": "XNA"})
        if "ORGANIZATION_TYPE" in cols:
            rules.append({"col": "ORGANIZATION_TYPE", "value": "XNA"})
        return rules

    @property
    def all_cols(self) -> list[str]:
        return self.numeric_cols + self.categorical_cols + self.list_cols + self.text_cols

    def source_tables(self, selected_secondary: list[str] | None = None) -> list[str]:
        tables = [self.primary_table]
        selected = selected_secondary if selected_secondary is not None else self.secondary_tables
        for name in selected:
            if name and name not in tables:
                tables.append(name)
        return tables

    def source_entries(self, selected_secondary: list[str] | None = None) -> list[dict]:
        return [
            {"id": f"s{idx}", "table": table}
            for idx, table in enumerate(self.source_tables(selected_secondary))
        ]

    def with_aux_tables(self, aux_tables: list[str]) -> "DataContext":
        selected = [t for t in aux_tables if t in self.secondary_tables]
        return replace(self, secondary_tables=selected)


def infer_data_context(data_name: str, data, task_type: str | None = None) -> DataContext:
    task_type = task_type or getattr(data, "task_type", None)
    if task_type == "rec":
        return _infer_rec_context(data_name, data)
    if task_type == "graph":
        return _infer_graph_context(data_name, data)
    return _infer_tabular_context(data_name, data)


def _profile_table(
    name: str,
    role: str,
    df,
    *,
    join_keys_to_primary: list[dict[str, str]] | None = None,
) -> TableContext:
    if df is None:
        return TableContext(name=name, role=role)
    columns = list(df.columns)
    return TableContext(
        name=name,
        role=role,
        shape=tuple(df.shape),
        dtypes={c: str(df[c].dtype) for c in columns},
        join_keys_to_primary=join_keys_to_primary or [],
    )


def _infer_time_col_from_columns(columns: list[str]) -> str | None:
    return _first_existing(
        columns,
        ("TransactionDT", "timestamp", "time", "ts", "time_step", "date", "datetime"),
    )


def _same_name_join_keys(primary_df, secondary_df, preferred: list[str]) -> list[dict[str, str]]:
    if primary_df is None or secondary_df is None:
        return []
    keys = []
    for key in preferred:
        if key and key in primary_df.columns and key in secondary_df.columns:
            keys.append({"primary": key, "secondary": key})
    if keys:
        return keys
    for key in primary_df.columns:
        if key in secondary_df.columns:
            keys.append({"primary": key, "secondary": key})
            break
    return keys


def _infer_rec_context(data_name: str, data) -> DataContext:
    interaction_df = data.interaction_df
    user_col = data._user_id_col
    item_col = data._item_id_col
    tables: dict[str, TableContext] = {
        "interaction": _profile_table("interaction", "primary", interaction_df)
    }
    secondary_tables = []
    if data.user_df is not None:
        secondary_tables.append("user")
        tables["user"] = _profile_table(
            "user",
            "secondary",
            data.user_df,
            join_keys_to_primary=[{"primary": user_col, "secondary": user_col}],
        )
    if data.item_df is not None:
        secondary_tables.append("item")
        tables["item"] = _profile_table(
            "item",
            "secondary",
            data.item_df,
            join_keys_to_primary=[{"primary": item_col, "secondary": item_col}],
        )
    return DataContext(
        task_type="rec",
        data_name=data_name,
        primary_table="interaction",
        secondary_tables=secondary_tables,
        tables=tables,
    )


def _infer_tabular_context(data_name: str, data) -> DataContext:
    train_df = data.train_df
    id_col = getattr(data, "id_col", None)
    aux_tables = {
        name: df
        for name, df in (getattr(data, "auxiliary_dfs", None) or {}).items()
        if df is not None
    }
    tables: dict[str, TableContext] = {
        "main": _profile_table("main", "primary", train_df)
    }
    preferred_keys = [id_col] if id_col else []
    for name, df in aux_tables.items():
        tables[name] = _profile_table(
            name,
            "secondary",
            df,
            join_keys_to_primary=_same_name_join_keys(train_df, df, preferred_keys),
        )
    return DataContext(
        task_type="tabular",
        data_name=data_name,
        primary_table="main",
        secondary_tables=list(aux_tables.keys()),
        tables=tables,
    )


def _infer_graph_context(data_name: str, data) -> DataContext:
    train_df = data.train_df
    id_col = getattr(data, "id_col", None)
    aux_tables = {
        name: df
        for name, df in (getattr(data, "auxiliary_dfs", None) or {}).items()
        if df is not None
    }
    tables: dict[str, TableContext] = {
        "main": _profile_table("main", "primary", train_df)
    }
    preferred_keys = [id_col] if id_col else []
    for name, df in aux_tables.items():
        tables[name] = _profile_table(
            name,
            "secondary",
            df,
            join_keys_to_primary=_same_name_join_keys(train_df, df, preferred_keys),
        )
    return DataContext(
        task_type="graph",
        data_name=data_name,
        primary_table="main",
        secondary_tables=list(aux_tables.keys()),
        tables=tables,
    )


def _is_numeric_dtype(dtype: str) -> bool:
    low = dtype.lower()
    return any(tok in low for tok in ("int", "float", "double")) and "datetime" not in low


def _first_existing(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_to_col = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower_to_col:
            return lower_to_col[cand.lower()]
    return None


def _special_cols(ctx: DataContext) -> set[str | None]:
    return {ctx.target_col, ctx.id_col, ctx.user_col, ctx.item_col, ctx.time_col}


def _dedupe(cols) -> list[str]:
    seen = set()
    out = []
    for col in cols:
        if col and col not in seen:
            seen.add(col)
            out.append(col)
    return out


__all__ = [
    "DataContext",
    "TableContext",
    "infer_data_context",
    "_infer_graph_context",
    "_infer_rec_context",
    "_infer_tabular_context",
]
