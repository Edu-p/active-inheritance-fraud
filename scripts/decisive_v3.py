"""Decisive v3 — smooth α-mixture curve (Fake.br raw OR ToLD-Br), 10 seeds.

Candidate pool(α) = α·synth + (1−α)·real-held-out-positive, at α ∈ {0,.25,.5,.75,1}.
Reuses decisive_v2.run_pool (dataset-agnostic).

Usage:
  uv run python scripts/decisive_v3.py --dataset fakebr-raw --learners minilm --seeds 10
  uv run python scripts/decisive_v3.py --dataset toldbr --learners minilm --seeds 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.embedder import SemanticEmbedder
from scripts.decisive_v2 import pool_auc, run_pool

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_data(ds):
    root = Path(f"dataset/{ds}")
    def csv(name):
        df = pd.read_csv(root / name)
        return df["preprocessed_news"].astype(str).tolist(), (df["label"] == "fake").astype(int).tolist()
    d0_texts, d0_y = csv("train-10pct.csv")
    va_texts, va_y = csv("validation.csv")
    te_texts, te_y = csv("test.csv")
    held_file = "heldout_fakes.csv" if ds == "fakebr-raw" else "heldout_toxic.csv"
    heldout = pd.read_csv(root / held_file).iloc[:, 0].astype(str).tolist()
    return d0_texts, d0_y, va_texts, va_y, te_texts, te_y, heldout


def load_synth(ds):
    if ds == "fakebr-raw":
        from scripts.shortcut_factorial import find_candidate_source
        src = find_candidate_source("fakebr-10pct", 42)
        meta = json.loads(src.read_text())
        return [t for b in meta["batches"] for t in b if t and t.strip()]
    # ToLD-Br: free-generated toxic pool
    pf = Path("outputs/decisive/toxic_synth.jsonl")
    if not pf.exists():
        return []
    return [json.loads(l)["text"] for l in pf.read_text().splitlines() if json.loads(l).get("text", "").strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fakebr-raw", choices=["fakebr-raw", "toldbr"])
    ap.add_argument("--learners", default="minilm,tfidf")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    d0_texts, d0_y, va_texts, va_y, te_texts, te_y, p0 = load_data(args.dataset)
    synth = load_synth(args.dataset)
    d0_fakes = [t for t, y in zip(d0_texts, d0_y) if y == 1]
    print(f"{args.dataset}: D0={len(d0_texts)} heldout={len(p0)} synth={len(synth)}", flush=True)

    rows = []
    for learner in [l for l in args.learners.split(",") if l.strip()]:
        if learner == "minilm":
            enc = SemanticEmbedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").model
            def E(ts):
                return enc.encode(ts, normalize_embeddings=True, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
            XR, XV, XT = E(d0_texts), E(va_texts), E(te_texts)
        for alpha in ALPHAS:
            n = 640
            n_synth = int(round(alpha * n))
            rng = np.random.default_rng(0)
            pool = list(rng.choice(synth, size=min(n_synth, len(synth)), replace=False)) + \
                   list(rng.choice(p0, size=min(n - n_synth, len(p0)), replace=False))
            if learner == "tfidf":
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True).fit(d0_texts + pool)
                XR, XV, XT = vec.transform(d0_texts), vec.transform(va_texts), vec.transform(te_texts)
                XC = vec.transform(pool)
            else:
                XC = E(pool)
            auc = pool_auc(d0_fakes, pool)
            r, rdf = run_pool(pool, XR, XV, XT, XC, d0_y, va_y, te_y, args.seeds)
            # bootstrap CI (paired by seed) for delta(All-Real) and delta(Score-Random)
            piv = rdf.pivot_table(index="seed", columns="rule", values="test_prauc")
            real_prauc = r["real_prauc"]
            def boot_ci(d, n=2000):
                d = np.asarray(d, dtype=float)
                d = d[~np.isnan(d)]
                if len(d) < 2:
                    return (float("nan"), float("nan"))
                rng_ = np.random.default_rng(0)
                ms = np.array([rng_.choice(d, len(d), replace=True).mean() for _ in range(n)])
                return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
            ci_ar = boot_ci(piv["All"].values - real_prauc)
            ci_sr = boot_ci(piv["Score"].values - piv["Random"].values)
            r.update(pool=f"mix{alpha:g}", learner=learner, auc_char=auc, n_pool=len(pool),
                     ci_all_real=ci_ar, ci_score_random=ci_sr)
            rows.append(r)
            print(f"mix{alpha:g}/{learner}: AUC={auc:.3f} | ΔAll-Real={r['delta_all_real']:+.4f} "
                  f"CI{tuple(round(x,4) for x in ci_ar)} | ΔScore-Rand={r['delta_score_random']:+.4f} "
                  f"CI{tuple(round(x,4) for x in ci_sr)}", flush=True)

    df = pd.DataFrame(rows)
    outdir = Path("outputs/decisive")
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"v3_{args.dataset}.csv", index=False)
    print("\n" + df.to_string(index=False) + f"\n-> outputs/decisive/v3_{args.dataset}.csv")


if __name__ == "__main__":
    main()
