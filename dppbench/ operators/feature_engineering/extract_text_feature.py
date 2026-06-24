import pandas as pd
from ..base_op import TextOp


class ExtractTextFeature(TextOp):
    """Extract BoW / TF-IDF / n-gram text features."""

    FIT_ON_TRAIN_ONLY = True

    def __init__(self, cols, method="tfidf", max_features=100, ngram_range=(1, 1)):
        super().__init__(name="ExtractTextFeature")
        if method not in ("tfidf", "bow"):
            raise ValueError("method must be tfidf or bow")
        self.cols = cols if isinstance(cols, list) else [cols]
        self.method = method
        self.max_features = int(max_features)
        self.ngram_range = tuple(ngram_range)
        self.vectorizers_ = {}
        self.fitted_ = False

    def get_op_description(self):
        description = """Operator name: ExtractTextFeature

Function description:
Extract sparse text statistics (TF-IDF or bag-of-words)
from text columns and append dense feature columns.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'review': ['good food', 'good service']})
>>> op = ExtractTextFeature(cols='review', method='bow', max_features=3)
>>> op.transform(df)
        review  review_bow_food  review_bow_good  review_bow_service
0    good food                1                1                   0
1 good service                0                1                   1

Example YAML:
  - op: ExtractTextFeature
    target: train
    params:
      cols: review
      method: tfidf
      max_features: 100
      ngram_range: [1, 2]
"""
        return description.strip()

    def transform(self, df):
        df = df.copy()
        try:
            if self.method == "tfidf":
                from sklearn.feature_extraction.text import TfidfVectorizer as Vec
            else:
                from sklearn.feature_extraction.text import CountVectorizer as Vec
        except Exception as exc:
            print(f"  [ExtractTextFeature] sklearn unavailable: {exc}")
            return df
        for col in self.cols:
            if col not in df.columns:
                continue
            text = df[col].fillna("").astype(str)
            if not self.fitted_ or col not in self.vectorizers_:
                self.vectorizers_[col] = Vec(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                ).fit(text)
            vec = self.vectorizers_[col]
            mat = vec.transform(text)
            names = vec.get_feature_names_out()
            base = f"{col}_{self.method}_"
            dense = mat.toarray()
            for i, name in enumerate(names):
                df[f"{base}{name}"] = dense[:, i]
        self.fitted_ = True
        return df
