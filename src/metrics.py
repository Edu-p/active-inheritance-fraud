"""Métricas das três dimensões da proposta."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def diversity_metrics(texts, embedder):
    """Similaridade média par-a-par (proxy de colapso modal)."""
    if len(texts) < 2:
        return {"n": len(texts), "mean_pairwise_sim": 1.0}
    X = embedder.transform(texts)
    sims = cosine_similarity(X)
    iu = np.triu_indices_from(sims, k=1)
    return {
        "n": len(texts),
        "mean_pairwise_sim": float(sims[iu].mean()),
    }
