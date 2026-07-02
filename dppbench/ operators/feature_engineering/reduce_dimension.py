import pandas as pd
from ..base_op import TabularOp


class ReduceDimension(TabularOp):
    """Reduce numeric feature dimensions with PCA/SVD/KPCA/LDA/UMAP."""

    FIT_ON_TRAIN_ONLY = True
    RANDOM_STATE = 42
    FILLNA = 0.0

    def __init__(self, cols=None, method="pca", n_components=8,
                 target_col=None, prefix=None, kernel="rbf", n_neighbors=15):
        super().__init__(name="ReduceDimension")
        if method not in ("pca", "svd", "kernel_pca", "lda", "umap"):
            raise ValueError("method must be pca/svd/kernel_pca/lda/umap")
        self.cols = cols
        self.method = method
        self.n_components = int(n_components)
        self.target_col = target_col
        self.prefix = prefix or f"{method}_"
        self.kernel = kernel
        self.n_neighbors = int(n_neighbors)
        self.cols_ = []
        self.reducer_ = None
        self.fitted_ = False

    def get_op_description(self):
        description = """Operator name: ReduceDimension

Function description:
Add low-dimensional numeric projections using PCA, SVD, Kernel PCA, LDA, or UMAP.
Optional dependencies gracefully no-op when absent.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : list[str] or None — Numeric columns to project. None = all numeric except target.
method : str — pca/svd/kernel_pca/lda/umap (default 'pca').
n_components : int — Output dimension (default 8).
target_col : str or None — Required for LDA.
prefix : str or None — Prefix for output columns. Default '{method}_'.
kernel : str — Kernel for kernel_pca (default 'rbf').
n_neighbors : int — Neighbours for UMAP (default 15).

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'x1': [1.0, 2.0], 'x2': [1.0, 0.0], 'label': [0, 1]})
>>> op = ReduceDimension(cols=['x1', 'x2'], n_components=1, method='pca', prefix='pc_')
>>> op.transform(df)
    x1   x2  label      pc_0
0  1.0  1.0      0 -0.707107
1  2.0  0.0      1  0.707107

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: ReduceDimension
    prev:
    - s0
    params:
      cols:
      - x1
      - x2
      - x3
      method: pca
      n_components: 2
      prefix: pc_
  train:
    prev:
    - o1
"""
        return description.strip()

    def _select_cols(self, df):
        if self.cols:
            return [c for c in self.cols if c in df.columns]
        return [
            c for c in df.select_dtypes(include=["number"]).columns
            if c != self.target_col
        ]

    def _make_reducer(self, n_components):
        if self.method == "pca":
            from sklearn.decomposition import PCA
            return PCA(n_components=n_components, random_state=self.RANDOM_STATE)
        if self.method == "svd":
            from sklearn.decomposition import TruncatedSVD
            return TruncatedSVD(n_components=n_components, random_state=self.RANDOM_STATE)
        if self.method == "kernel_pca":
            from sklearn.decomposition import KernelPCA
            return KernelPCA(
                n_components=n_components,
                kernel=self.kernel,
                random_state=self.RANDOM_STATE,
            )
        if self.method == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            return LinearDiscriminantAnalysis(n_components=n_components)
        import umap
        return umap.UMAP(
            n_components=n_components,
            n_neighbors=self.n_neighbors,
            random_state=self.RANDOM_STATE,
        )

    def transform(self, df):
        df = df.copy()
        if not self.fitted_:
            self.cols_ = self._select_cols(df)
        cols = [c for c in self.cols_ if c in df.columns]
        if not cols:
            return df
        x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(self.FILLNA)
        n_components = min(self.n_components, len(cols), max(1, len(df)))
        y = None
        if self.method == "lda":
            if self.target_col not in df.columns:
                return df
            y = df[self.target_col]
            n_components = min(n_components, max(1, y.nunique(dropna=True) - 1))
        try:
            if not self.fitted_:
                self.reducer_ = self._make_reducer(n_components)
                projected = self.reducer_.fit_transform(x, y) if y is not None else self.reducer_.fit_transform(x)
                self.fitted_ = True
            else:
                projected = self.reducer_.transform(x)
        except Exception as exc:
            print(f"  [ReduceDimension] {self.method} unavailable/failed: {exc}")
            return df
        for i in range(projected.shape[1]):
            df[f"{self.prefix}{i}"] = projected[:, i]
        return df
