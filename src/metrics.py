"""Metrics for the three dimensions of the proposal."""
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
    """Mean pairwise similarity (modal-collapse proxy)."""
    if len(texts) < 2:
        return {"n": len(texts), "mean_pairwise_sim": 1.0}
    X = embedder.transform(texts)
    sims = cosine_similarity(X)
    iu = np.triu_indices_from(sims, k=1)
    return {
        "n": len(texts),
        "mean_pairwise_sim": float(np.clip(sims[iu].mean(), 0.0, 1.0)),
    }


def coverage_metrics(texts, n: int = 3):
    """Distinct n-gram ratio — proxy de cobertura / long tail."""
    grams = set()
    total = 0
    for t in texts:
        toks = t.split()
        if len(toks) >= n:
            for i in range(len(toks) - n + 1):
                grams.add(" ".join(toks[i : i + n]))
                total += 1
    return {
        "distinct_ngrams": len(grams),
        "total_ngrams": total,
        "ngram_ttr": float(len(grams) / total) if total else 0.0,
    }


def wasserstein_fidelity(real_emb, synth_emb, n_components: int = 3, seed: int = 42):
    """1D Wasserstein averaged over the top-k PCA components of the real data."""
    from scipy.sparse import issparse
    from scipy.stats import wasserstein_distance
    from sklearn.decomposition import PCA

    real = real_emb.toarray() if issparse(real_emb) else np.asarray(real_emb)
    synth = synth_emb.toarray() if issparse(synth_emb) else np.asarray(synth_emb)
    if real.ndim != 2 or synth.ndim != 2 or len(real) < 2 or len(synth) < 2 or real.shape[1] < 2:
        return {"wasserstein": None}

    k = min(n_components, real.shape[1], len(real))
    pca = PCA(n_components=k, random_state=seed).fit(real)
    r = pca.transform(real)
    s = pca.transform(synth)
    ds = [wasserstein_distance(r[:, i], s[:, i]) for i in range(k)]
    return {"wasserstein": float(np.mean(ds))}


def synthetic_quality(embedder, real_texts, synth_texts):
    """Diversidade + cobertura + fidelidade (Wasserstein) de um conjunto sintético."""
    out = {"diversity": diversity_metrics(synth_texts, embedder)}
    out["coverage"] = coverage_metrics(synth_texts)
    if synth_texts and len(real_texts) >= 2:
        out["fidelity"] = wasserstein_fidelity(
            embedder.transform(real_texts), embedder.transform(synth_texts)
        )
    else:
        out["fidelity"] = {"wasserstein": None}
    return out
