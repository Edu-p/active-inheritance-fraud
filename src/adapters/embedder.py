"""Text embedders. Semantic (default) ou TF-IDF (rápido, p/ sanity checks)."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    def __init__(self, max_features: int = 10_000):
        self.max_features = max_features
        self.vec = TfidfVectorizer(max_features=max_features, sublinear_tf=True)

    def fit(self, texts):
        self.vec.fit(texts)
        return self

    def transform(self, texts):
        return self.vec.transform(texts)


class SemanticEmbedder:
    """Sentence-transformer multilíngue; `fit` é no-op (mantém a interface)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        return self

    def transform(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
