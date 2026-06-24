from ..base_op import TabularOp


class CorrectTypo(TabularOp):
    """Correct common text typos using exact mapping or fuzzy matching."""

    def __init__(self, cols, mapping=None, vocabulary=None, threshold=90):
        super().__init__(name="CorrectTypo")
        self.cols = cols if isinstance(cols, list) else [cols]
        self.mapping = mapping or {}
        self.vocabulary = vocabulary or []
        self.threshold = int(threshold)

    def get_op_description(self):
        description = """Operator name: CorrectTypo

Function description:
Correct misspellings in text columns. Applies explicit mapping first; if
rapidfuzz is installed and vocabulary is supplied, performs fuzzy
nearest-vocabulary correction above threshold. Replacement is in place.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : str/list[str] — Text columns.
mapping : dict — Exact replacement map.
vocabulary : list[str] — Valid terms for fuzzy matching.
threshold : int — Fuzzy score cutoff.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'city': ['Beijing', 'Shanghi', 'Shenzen']})
>>> op = CorrectTypo(cols='city', mapping={'Shanghi': 'Shanghai', 'Shenzen': 'Shenzhen'})
>>> op.transform(df)
       city
0   Beijing
1  Shanghai
2  Shenzhen

Example YAML:
  - op: CorrectTypo
    target: both
    params:
      cols: city
      mapping:
        Shanghi: Shanghai
        Shenzen: Shenzhen
"""
        return description.strip()

    def _correct_value(self, value):
        if value in self.mapping:
            return self.mapping[value]
        if not self.vocabulary or value is None:
            return value
        try:
            from rapidfuzz import process, fuzz
            match = process.extractOne(str(value), self.vocabulary, scorer=fuzz.ratio)
            if match and match[1] >= self.threshold:
                return match[0]
        except Exception:
            return value
        return value

    def transform(self, df):
        df = df.copy()
        for col in self.cols:
            if col not in df.columns:
                continue
            df[col] = df[col].apply(self._correct_value)
        return df
