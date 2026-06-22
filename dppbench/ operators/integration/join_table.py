import pandas as pd
from ..base_op import TabularOp


class JoinTable(TabularOp):
    """Join one or more auxiliary tables onto the main table.

    Unifies the old MergeOnKey (1:1 key join), AggMerge (group-by aggregate
    then join) and RecJoin (recsys user/item double join) operators via the
    ``method`` switch.

      - method="key" : left/inner-merge ``aux_df`` on ``key_col`` (1:1 lookup).
      - method="agg" : group ``aux_df`` by ``key_col``, aggregate numeric cols
        with ``agg_funcs`` (categorical -> nunique), then merge.
      - method="rec" : merge ``user_df`` on ``user_col`` and ``item_df`` on
        ``item_col`` (recommendation interaction enrichment).
    """

    def __init__(self, method="key",
                 aux_df=None, key_col=None, how="left",
                 suffixes=("", "_aux"),
                 agg_funcs=None, prefix=None, max_cols=None,
                 user_col="user_id", item_col="item_id",
                 user_df=None, item_df=None):
        super().__init__(name="JoinTable")
        if method not in ("key", "agg", "rec"):
            raise ValueError(
                f"JoinTable: method must be 'key'/'agg'/'rec', got '{method}'"
            )
        self.method = method
        self.aux_df = aux_df
        self.key_col = key_col
        self.how = how
        self.suffixes = tuple(suffixes) if suffixes is not None else ("", "_aux")
        self.agg_funcs = agg_funcs or ["mean", "max", "min"]
        self.prefix = prefix
        self.max_cols = max_cols
        self.user_col = user_col
        self.item_col = item_col
        self.user_df = user_df
        self.item_df = item_df

    def get_op_description(self):
        description = """Operator name: JoinTable

Function description:
Horizontally merge one or more tables onto the main
table by key. Three modes via ``method``:
- 'key' : 1:1 left/inner merge of ``aux_df`` on ``key_col`` (lookup join).
- 'agg' : group ``aux_df`` by ``key_col`` and aggregate numeric columns
(``agg_funcs``) / categorical columns (nunique), then merge.
- 'rec' : merge ``user_df`` on ``user_col`` and ``item_df`` on ``item_col``.

Input:
df : pd.DataFrame — Main / interaction table.

Parameters:
method : str — 'key' (default) / 'agg' / 'rec'.
aux_df : pd.DataFrame — Auxiliary table (key/agg modes; $name in YAML).
key_col : str — Join/group key (key/agg modes).
how : str — Merge strategy. Default 'left'.
suffixes : tuple[str,str] — Overlap suffixes (key mode). Default ("", "_aux").
agg_funcs : list[str] — Numeric aggregations (agg mode). Default [mean,max,min].
prefix : str — Column prefix for aggregated features (agg mode).
max_cols : int or None — Cap on aggregated columns (agg mode).
user_col / item_col : str — Join keys (rec mode).
user_df / item_df : pd.DataFrame — Side tables (rec mode; $name in YAML).

Output:
pd.DataFrame — Main table enriched with joined columns.

Example:
>>> df = pd.DataFrame({'txId': [1, 2, 3]})
>>> aux = pd.DataFrame({'txId': [1, 2, 3], 'label': ['A', 'B', 'C']})
>>> op = JoinTable(method='key', aux_df=aux, key_col='txId')
>>> op.transform(df)
   txId label
0     1     A
1     2     B
2     3     C

Example YAML:
  - op: JoinTable
    target: both
    params:
      method: agg
      aux_df: $bureau
      key_col: SK_ID_CURR
      prefix: BUREAU
      max_cols: 20
"""
        return description.strip()

    def transform(self, df):
        if self.method == "rec":
            return self._transform_rec(df)
        if self.method == "agg":
            return self._transform_agg(df)
        return self._transform_key(df)

    def _transform_key(self, df):
        aux = self.aux_df
        if aux is None or len(aux) == 0:
            return df
        if self.key_col not in df.columns or self.key_col not in aux.columns:
            return df
        df = df.copy()
        aux = aux.copy()
        tmp_key = "__dppbench_merge_key__"
        while tmp_key in df.columns or tmp_key in aux.columns:
            tmp_key = f"_{tmp_key}"
        left_missing = "__DPPBENCH_LEFT_MISSING__"
        right_missing = "__DPPBENCH_RIGHT_MISSING__"
        df[tmp_key] = df[self.key_col].where(
            df[self.key_col].notna(), left_missing
        ).astype(str)
        aux[tmp_key] = aux[self.key_col].where(
            aux[self.key_col].notna(), right_missing
        ).astype(str)
        overlap_cols = [c for c in aux.columns if c != self.key_col and c in df.columns]
        aux = aux.drop(columns=[self.key_col])
        result = df.merge(aux, on=tmp_key, how=self.how, suffixes=self.suffixes)
        result = result.drop(columns=[tmp_key])
        right_suffix = self.suffixes[1] if len(self.suffixes) > 1 else "_aux"
        for col in overlap_cols:
            aux_col = f"{col}{right_suffix}"
            if aux_col not in result.columns:
                continue
            result[col] = result[col].combine_first(result[aux_col])
            result = result.drop(columns=[aux_col])
        return result

    def _transform_agg(self, df):
        aux = self.aux_df
        if aux is None or len(aux) == 0:
            return df
        if self.key_col not in df.columns or self.key_col not in aux.columns:
            return df
        num_cols = [c for c in aux.columns
                    if c != self.key_col and aux[c].dtype.kind in ("i", "u", "f")]
        cat_cols = [c for c in aux.columns
                    if c != self.key_col and aux[c].dtype.kind in ("O", "b")]
        if self.max_cols is not None:
            num_cols = num_cols[:self.max_cols]
            cat_cols = cat_cols[:self.max_cols]
        agg_dict = {}
        for col in num_cols:
            agg_dict[col] = self.agg_funcs
        for col in cat_cols:
            agg_dict[col] = ["nunique"]
        if not agg_dict:
            return df
        grouped = aux.groupby(self.key_col).agg(agg_dict)
        grouped.columns = [
            f"{self.prefix}_{col}_{func}" if self.prefix else f"{col}_{func}"
            for col, func in grouped.columns
        ]
        grouped = grouped.reset_index()
        return df.merge(grouped, on=self.key_col, how="left")

    def _transform_rec(self, df):
        result = df.copy()
        if self.user_df is not None:
            result = pd.merge(result, self.user_df, on=self.user_col, how=self.how)
        if self.item_df is not None:
            result = pd.merge(result, self.item_df, on=self.item_col, how=self.how)
        return result
