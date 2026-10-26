"""Helpers de aplicação compartilhados (avaliação + ajuste de limiar).

Elimina a duplicação que existia entre main.py, baseline.py e pipeline.py.
"""
from __future__ import annotations

from ..domain.metrics import best_threshold, classification_metrics, probabilistic_metrics


def evaluate(learner, texts, y, threshold=None):
    """Métricas de classificação + probabilísticas + limiar usado."""
    proba = learner.predict(texts)[:, 1]
    thr = threshold if threshold is not None else 0.5
    pred = (proba >= thr).astype(int)
    out = classification_metrics(y, pred)
    out.update(probabilistic_metrics(y, proba))
    out["threshold"] = float(thr)
    return out


def tune_threshold(learner, val):
    """Melhor limiar F1 na validação (val = (texts, y))."""
    return best_threshold(val[1], learner.predict(val[0])[:, 1])
