"""Decisive v2 — graduated detectability with REAL LLM text, raw Fake.br.

Candidate pools (content ~fixed, only the voice/register changes):
  p0     raw held-out real fakes            (AUC ~0.5)
  p1     light paraphrase (same claims)     (AUC medium)
  p2     generalized names (generic rumor)  (AUC higher)
  p3     "circulating rumor" register       (AUC high)
  synth  free LLM generation (raw deepseek) (AUC ~1.0)

Runs the GSS loop (T=5,b=64,k=16; Real/All/Random/Score/Easy) with TF-IDF+LR and
MiniLM+LR learners, reports pool detectability AUC (char 3-5) and
delta(All-Real), delta(Score-Random), delta(Easy-Random).

CPU only. Zero API (uses outputs/decisive/paraphrases.jsonl).

Usage:
  uv run python scripts/decisive_v2.py --pools p0,p1,p2,p3,synth --learners minilm,tfidf --seeds 5
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

from src.embedder import SemanticEmbedder

T, B, K = 5, 64, 16
W = (0.4, 0.4, 0.2)
H2 = lambda p: -(p * np.log2(np.clip(p, 1e-12, 1)) + (1 - p) * np.log2(np.clip(1 - p, 1e-12, 1)))
ROOT = Path("dataset/fakebr-raw")


def load_csv(name):
    df = pd.read_csv(ROOT / name)
    return df["preprocessed_news"].astype(str).tolist(), (df["label"] == "fake").astype(int).tolist()


def load_pools():
    heldout = pd.read_csv(ROOT / "heldout_fakes.csv")["preprocessed_news"].astype(str).tolist()
    par = {lvl: [] for lvl in ("p1", "p2", "p3")}
    pf = Path("outputs/decisive/paraphrases.jsonl")
    if pf.exists():
        for line in pf.read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("level") in par and d.get("text", "").strip():
                par[d["level"]].append(d["text"].strip())
    from scripts.shortcut_factorial import find_candidate_source
    src = find_candidate_source("fakebr-10pct", 42)
    meta = json.loads(src.read_text())
    synth = [t for b in meta["batches"] for t in b if t and t.strip()]
    return {"p0": heldout, "p1": par["p1"], "p2": par["p2"], "p3": par["p3"], "synth": synth}


def pool_auc(d0_fakes, cand_texts):
    if len(cand_texts) < 10:
        return float("nan")
    texts = np.array(d0_fakes + cand_texts, dtype=object)
    y = np.r_[np.zeros(len(d0_fakes)), np.ones(len(cand_texts))]
    p = cross_val_predict(
        make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
                      LogisticRegression(C=1, max_iter=3000, class_weight="balanced")),
        texts, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def vstack(a, b):
    return sparse.vstack([a, b]).tocsr() if sparse.issparse(a) else np.vstack([a, b])


def run_pool(pool_texts, XR, XV, XT, XC, d0_y, va_y, te_y, seeds):
    n_pool = T * B
    d0_fakes = []  # filled below is not needed here; pool_auc needs d0 fakes, passed separately
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        n = min(n_pool, len(pool_texts))
        idx = rng.permutation(len(pool_texts))[:n]
        XCp = XC[idx]  # n rows; H below indexes positions 0..n-1

        def fit(H, s):
            X, y = XR, list(d0_y)
            if H:
                X = vstack(X, XCp[H])
                y = y + [1] * len(H)
            return LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=s).fit(X, y)

        def ev(m):
            pv = m.predict_proba(XV)[:, 1]
            pt = m.predict_proba(XT)[:, 1]
            pr, rc, th = precision_recall_curve(va_y, pv)
            f = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
            thr = th[np.argmax(f[:-1])]
            return dict(test_prauc=average_precision_score(te_y, pt), test_f1=f1_score(te_y, pt >= thr))

        def score(m, cidx, H, first):
            p = m.predict_proba(XCp[cidx])[:, 1]
            e = -np.log(np.clip(p, 1e-12, 1))
            e = (e - e.min()) / (e.max() - e.min()) if e.max() > e.min() else np.zeros_like(e)
            u = H2(p)
            if first or not H:
                d = np.ones(len(cidx))
            else:
                A = XCp[cidx] @ XCp[H].T
                A = A.toarray() if sparse.issparse(A) else A
                d = 1 - np.clip(A.max(axis=1), 0, 1)
            return W[0] * e + W[1] * u + W[2] * d

        batches = [np.arange(t * B, (t + 1) * B) for t in range(T)]

        def rec(rule, H):
            m = fit(H, seed)
            r = ev(m)
            r.update(rule=rule, n_H=len(H))
            rows.append(r)

        rec("All", list(range(n)))
        Hr = []
        for bt in batches:
            Hr += list(rng.choice(bt, K, replace=False))
        rec("Random", Hr)
        for rule, sign in (("Score", +1), ("Easy", -1)):
            H = []
            for t, bt in enumerate(batches):
                m = fit(H, seed)
                s = score(m, bt, H, t == 0)
                H += list(bt[np.argsort(-sign * s)[:K]])
            rec(rule, H)

    # Real (no synthetic)
    m0 = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(XR, d0_y)
    pv = m0.predict_proba(XV)[:, 1]
    pt = m0.predict_proba(XT)[:, 1]
    pr, rc, th = precision_recall_curve(va_y, pv)
    f = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    real_prauc = average_precision_score(te_y, pt)

    df = pd.DataFrame(rows)
    out = {}
    for rule in ("All", "Random", "Score", "Easy"):
        sub = df[df.rule == rule]
        out[f"prauc_{rule}"] = float(sub.test_prauc.mean()) if len(sub) else float("nan")
    out["real_prauc"] = float(real_prauc)
    out["delta_all_real"] = out["prauc_All"] - out["real_prauc"]
    out["delta_score_random"] = out["prauc_Score"] - out["prauc_Random"]
    out["delta_easy_random"] = out["prauc_Easy"] - out["prauc_Random"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="p0,p1,p2,p3,synth")
    ap.add_argument("--learners", default="minilm,tfidf")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    d0_texts, d0_y = load_csv("train-10pct.csv")
    va_texts, va_y = load_csv("validation.csv")
    te_texts, te_y = load_csv("test.csv")
    pools = load_pools()
    d0_fakes = [t for t, y in zip(d0_texts, d0_y) if y == 1]

    for learner in [l for l in args.learners.split(",") if l.strip()]:
        if learner == "minilm":
            enc = SemanticEmbedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").model
            def E(ts):
                return enc.encode(ts, normalize_embeddings=True, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
            XR, XV, XT = E(d0_texts), E(va_texts), E(te_texts)
        for pname in [p for p in args.pools.split(",") if p.strip()]:
            pool = pools.get(pname, [])
            if not pool:
                print(f"SKIP {pname}/{learner} (empty)", flush=True)
                continue
            pool = pool[: 640]
            if learner == "tfidf":
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True).fit(d0_texts + pool)
                XR, XV, XT = vec.transform(d0_texts), vec.transform(va_texts), vec.transform(te_texts)
                XC = vec.transform(pool)
            else:
                XC = E(pool)
            auc = pool_auc(d0_fakes, pool)
            r = run_pool(pool, XR, XV, XT, XC, d0_y, va_y, te_y, args.seeds)
            r["pool"] = pname
            r["learner"] = learner
            r["auc_char"] = auc
            r["n_pool"] = len(pool)
            print(f"{pname}/{learner}: AUC={auc:.3f} | ΔAll-Real={r['delta_all_real']:+.4f} "
                  f"ΔScore-Rand={r['delta_score_random']:+.4f} ΔEasy-Rand={r['delta_easy_random']:+.4f}", flush=True)
            outdir = Path("outputs/decisive")
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / f"v2_{pname}_{learner}.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
