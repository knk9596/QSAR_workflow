"""Custom scikit-learn transformers."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CorrPrune(BaseEstimator, TransformerMixin):
    """Drop one column from every pair correlated above ``thresh``.

    Deterministic: always keeps the lower-indexed column of a pair. Fitted
    per-fold so no test-fold information reaches the reduction.
    """

    def __init__(self, thresh: float = 0.95):
        self.thresh = thresh

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        corr = np.nan_to_num(np.corrcoef(X, rowvar=False))
        n = corr.shape[0]
        dropped: set[int] = set()
        for i in range(n):
            if i in dropped:
                continue
            for j in range(i + 1, n):
                if j not in dropped and abs(corr[i, j]) > self.thresh:
                    dropped.add(j)
        self.keep_ = np.array([i for i in range(n) if i not in dropped], dtype=int)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in self.keep_])
        return np.asarray(input_features)[self.keep_]
