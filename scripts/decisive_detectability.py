"""Decisive experiment v1 — graduated detectability via alpha-mixture (zero API).

For a scenario, builds candidate pools that interpolate between REAL held-out
fakes (indistinguishable, AUC ~0.5) and frozen LLM synthetic (AUC ~1.0):

    pool(alpha) = alpha*synthetic + (1-alpha)*held_out_real_fakes

runs the GSS loop (T=5, b=64, k=16; Real/All/Random/Score/Easy) with a cheap
TF-IDF+LR learner, and reports, per alpha:
    pool detectability AUC (char 3-5 gram detector, 5-fold CV)
    test PR-AUC per rule + delta(All-Real), delta(Score-Random), delta(Easy-Random)

CPU only. Reuses the advisor's pilot loop design (docs/.../04_piloto_llm/loop.py),
adapted to the repo's frozen candidates.

Usage:
  uv run python scripts/decisive_detectability.py --scenario fakebr-10pct --alphas 0,0.5,1 --seeds 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from scripts.controlled_curatorship import SCENARIOS, paths

T, B, K = 5, 64, 16
W = (0.4, 0.4, 0.2)
H2 = lambda p: -(p * np.log2(np.clip(p, 1e-12, 1)) + (1 - p) * np.log2(np.clip(1 - p, 1e-12, 1)))


def load_real(scenario):
    csv, val_csv, test_csv, _ = paths(scenario)
    tr = pd.read_csv(csv)
    va = pd.read_csv(val_csv)
    te = pd.read_csv(test_csv)
    return (tr["preprocessed_news"].astype(str).tolist(), (tr["label"] == "fake").astype(int).tolist(),
            va["preprocessed_news"].astype(str).tolist(), (va["label"] == "fake").astype(int).tolist(),
            te["preprocessed_news"].astype(str).tolist(), (te["label"] == "fake").astype(int).tolist())


def held_out_real_fakes(scenario, d0_texts):
    """Real fakes NOT in D0 (the 90% dropped by scarcity), to use as alpha=0 candidates."""
    full = pd.read_csv("dataset/pre-processed.csv")
    full_fake = [t for t, y in zip(full["preprocessed_news"].astype(str), (full["label"] == "fake")) if y]
    d0 = set(d0_texts)
    return list(dict.fromkeys(t for t in full_fake if t not in d0))


def find_candidates(scenario, seed):
    from scripts.shortcut_factorial import find_candidate_source
    src = find_candidate_source(scenario, seed)
    meta = json.loads(src.read_text())
    batches = meta["batches"]
    out = []
    for b in batches:
        out.extend([t for t in b if t and t.strip()])
    return out


def build_pool(synth, real_fakes, alpha, n, rng):
    n_synth = int(round(alpha * n))
    n_real = n - n_synth
    pool = list(rng.choice(real_fakes, size=min(n_real, len(real_fakes)), replace=False))
    pool += list(rng.choice(synth, size=min(n_synth, len(synth)), replace=False))
    return pool


def fit_learner(vec, XR, y_tr, XC, H, seed):
    X, y = XR, list(y_tr)
    if len(H):
        X = sparse.vstack([XR, XC[H]]).tocsr()
        y = y + [1] * len(H)
    return LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=seed).fit(X, y)


def evaluate(m, vec, va_texts, va_y, te_texts, te_y):
    Xv = vec.transform(va_texts)
    Xt = vec.transform(te_texts)
    pv = m.predict_proba(Xv)[:, 1]
    pt = m.predict_proba(Xt)[:, 1]
    pr, rc, th = precision_recall_curve(va_y, pv)
    f = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    thr = th[np.argmax(f[:-1])]
    return dict(test_prauc=average_precision_score(te_y, pt),
                test_f1=f1_score(te_y, pt >= thr), thr=thr)


def score_candidates(m, XC, cidx, H, first):
    p = m.predict_proba(XC[cidx])[:, 1]
    e = -np.log(np.clip(p, 1e-12, 1))
    e = (e - e.min()) / (e.max() - e.min()) if e.max() > e.min() else np.zeros_like(e)
    u = H2(p)
    if first or not H:
        d = np.ones(len(cidx))
    else:
        A = XC[cidx] @ XC[H].T
        d = 1 - np.asarray(A.max(axis=1).todense()).ravel() if sparse.issparse(A) else 1 - A.max(axis=1)
    return W[0] * e + W[1] * u + W[2] * d


def pool_auc(vec_det, d0_fakes, cand_texts):
    """Detectability AUC of the pool vs real fakes (char 3-5 TF-IDF+LR, 5-fold CV)."""
    if len(cand_texts) < 10:
        return float("nan")
    texts = np.array(d0_fakes + cand_texts, dtype=object)
    y = np.r_[np.zeros(len(d0_fakes)), np.ones(len(cand_texts))]
    p = cross_val_predict(
        make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
                      LogisticRegression(C=1, max_iter=3000, class_weight="balanced")),
        texts, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba",
    )[:, 1]
    return float(roc_auc_score(y, p))


def run_alpha(scenario, alpha, synth, real_fakes, d0_texts, d0_y, va_texts, va_y, te_texts, te_y, seeds):
    n_pool = T * B
    d0_fakes = [t for t, y in zip(d0_texts, d0_y) if y == 1]
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        pool = build_pool(synth, real_fakes, alpha, n_pool, rng)
        # fit vectorizer on D0 + pool (transductive in candidates, as the pilot)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True).fit(d0_texts + pool)
        XR = vec.transform(d0_texts)
        XC = vec.transform(pool)
        # order candidates into 5 batches of 64
        order = rng.permutation(len(pool))[: n_pool]
        batches = [order[t * B:(t + 1) * B] for t in range(T)]
        auc = pool_auc(None, d0_fakes, pool)

        def rec(rule, H):
            m = fit_learner(vec, XR, d0_y, XC, H, seed)
            r = evaluate(m, vec, va_texts, va_y, te_texts, te_y)
            r.update(rule=rule, n_H=len(H))
            rows.append(r)

        rec("All", list(order))
        Hr = []
        for bt in batches:
            Hr += list(rng.choice(bt, K, replace=False))
        rec("Random", Hr)
        # Score / Easy (warm learner)
        for rule, sign in (("Score", +1), ("Easy", -1)):
            H = []
            for t, bt in enumerate(batches):
                m = fit_learner(vec, XR, d0_y, XC, H, seed)
                s = score_candidates(m, XC, bt, H, t == 0)
                sel = bt[np.argsort(-sign * s)[:K]]
                H += list(sel)
            rec(rule, H)
    # Real (no synthetic)
    vec0 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True).fit(d0_texts)
    m0 = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(vec0.transform(d0_texts), d0_y)
    r0 = evaluate(m0, vec0, va_texts, va_y, te_texts, te_y)
    r0.update(rule="Real", n_H=0)

    df = pd.DataFrame(rows)
    out = {"alpha": alpha, "pool_auc": auc}
    for rule in ("All", "Random", "Score", "Easy"):
        sub = df[df.rule == rule]
        out[f"prauc_{rule}"] = float(sub.test_prauc.mean()) if len(sub) else float("nan")
    out["real_prauc"] = float(r0["test_prauc"])
    out["delta_all_real"] = out["prauc_All"] - out["real_prauc"]
    out["delta_score_random"] = out["prauc_Score"] - out["prauc_Random"]
    out["delta_easy_random"] = out["prauc_Easy"] - out["prauc_Random"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="fakebr-10pct")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    d0_texts, d0_y, va_texts, va_y, te_texts, te_y = load_real(args.scenario)
    real_fakes = held_out_real_fakes(args.scenario, d0_texts)
    synth = find_candidates(args.scenario, 42)  # frozen deepseek pool (one seed's set)
    print(f"real fakes held out: {len(real_fakes)} | synthetic pool: {len(synth)}")

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    rows = []
    for alpha in alphas:
        r = run_alpha(args.scenario, alpha, synth, real_fakes, d0_texts, d0_y, va_texts, va_y, te_texts, te_y, args.seeds)
        rows.append(r)
        print(f"alpha={alpha:.2f} AUC={r['pool_auc']:.3f} | "
              f"ΔAll-Real={r['delta_all_real']:+.4f} ΔScore-Rand={r['delta_score_random']:+.4f} "
              f"ΔEasy-Rand={r['delta_easy_random']:+.4f}", flush=True)

    outdir = Path("outputs/decisive")
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    dest = outdir / f"{args.scenario}.csv"
    df.to_csv(dest, index=False)
    print(f"\n{df.to_string(index=False)}\n-> {dest}")


if __name__ == "__main__":
    main()
