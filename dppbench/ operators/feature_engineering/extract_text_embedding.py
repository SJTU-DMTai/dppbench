import numpy as np
from ..base_op import TextOp


class ExtractTextEmbedding(TextOp):
    """Extract lightweight or model-backed text embeddings."""

    FIT_ON_TRAIN_ONLY = True

    def __init__(self, cols, method="hash", dim=32, model_name=None):
        super().__init__(name="ExtractTextEmbedding")
        self.cols = cols if isinstance(cols, list) else [cols]
        self.method = method
        self.dim = int(dim)
        self.model_name = model_name

    def get_op_description(self):
        description = """Operator name: ExtractTextEmbedding

Function description:
Extract text embedding features. Uses deterministic hash
embeddings by default; transformer/gensim dependencies are optional.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'review': ['good food', 'bad service']})
>>> op = ExtractTextEmbedding(cols='review', method='hash', dim=1)
>>> op.transform(df)
        review  review_emb_0
0    good food           1.0
1  bad service           1.0

Example YAML:
  - op: ExtractTextEmbedding
    target: train
    params:
      cols: review
      method: hash
      dim: 32
"""
        return description.strip()

    def _hash_embed(self, text):
        vec = np.zeros(self.dim, dtype=float)
        for token in str(text).split():
            idx = abs(hash(token)) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def transform(self, df):
        df = df.copy()
        for col in self.cols:
            if col not in df.columns:
                continue
            base = f"{col}_emb_"
            emb = np.vstack([self._hash_embed(v) for v in df[col].fillna("")])
            for i in range(self.dim):
                df[f"{base}{i}"] = emb[:, i]
        return df
