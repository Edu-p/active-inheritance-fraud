"""Adversarial evaluator: scores synthetic samples for inheritance.

score(x) = w_err·error + w_unc·uncertainty + w_div·diversity
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    return np.ones_like(x) if span < 1e-12 else (x - x.min()) / span


class AdversarialEvaluator:
    def __init__(
        self,
        learner,
        embedder,
        w_error: float = 0.4,
        w_uncertainty: float = 0.4,
        w_diversity: float = 0.2,
    ) -> None:
        self.learner = learner
        self.embedder = embedder
        self.w_error = w_error
        self.w_uncertainty = w_uncertainty
        self.w_diversity = w_diversity

    def diversity(self, texts: list[str], inherited_texts: list[str]) -> np.ndarray:
        """1 - max cos-sim against the inherited pool → [0, 1]."""
        if not inherited_texts:
            return np.ones(len(texts))
        X = self.embedder.transform(texts)
        P = self.embedder.transform(inherited_texts)
        return 1.0 - cosine_similarity(X, P).max(axis=1)

    def score(
        self,
        texts: list[str],
        y_synth: list[int],
        inherited_texts: list[str],
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Per-sample multi-objective score + component averages."""
        err = _minmax(self.learner.error(texts, y_synth))
        unc = self.learner.compute_uncertainty(texts)
        div = self.diversity(texts, inherited_texts)
        total = self.w_error * err + self.w_uncertainty * unc + self.w_diversity * div
        parts = {
            "error": float(err.mean()),
            "uncertainty": float(unc.mean()),
            "diversity": float(div.mean()),
        }
        return total, parts

    def select(
        self,
        texts: list[str],
        y_synth: list[int],
        inherited_texts: list[str],
        k: int,
    ) -> tuple[list[str], float, dict[str, float]]:
        """Top-K samples that maximize the multi-objective score."""
        total, parts = self.score(texts, y_synth, inherited_texts)
        order = np.argsort(-total)
        selected = [texts[i] for i in order[:k]]
        return selected, float(total[order[:k]].mean()), parts
