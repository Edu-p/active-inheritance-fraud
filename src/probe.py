"""Frozen multilingual text encoder + logistic decision probe (NOT a JEPA/JEV model)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class SemanticProbe:
    def __init__(self, embedder):
        self.embedder = embedder
        self.cache = {}
        self.model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)

    def _vectors(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            for text, vector in zip(missing, self.embedder.transform(missing)):
                self.cache[text] = vector
        return np.asarray([self.cache[t] for t in texts])

    def fit(self, texts, y, epochs=3):
        self.model.fit(self._vectors(texts), y)
        probs = self.predict(texts)
        return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())

    def predict(self, texts):
        return self.model.predict_proba(self._vectors(texts))

    def predict_class(self, texts):
        return self.predict(texts).argmax(axis=1)

    def compute_uncertainty(self, texts):
        p = np.clip(self.predict(texts), 1e-12, 1)
        return -(p * np.log(p)).sum(axis=1) / np.log(2)

    def error(self, texts, y):
        p = self.predict(texts)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
