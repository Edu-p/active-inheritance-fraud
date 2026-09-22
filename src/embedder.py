"""Embedders de texto. TF-IDF é o padrão (sem dependências pesadas)."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    def __init__(self, max_features: int = 10_000):
        self.max_features = max_features
        self.vec = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=True,
        )

    def fit(self, texts):
        self.vec.fit(texts)
        return self

    def transform(self, texts):
        return self.vec.transform(texts)
