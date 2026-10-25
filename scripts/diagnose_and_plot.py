"""Offline diagnosis (real-vs-synthetic) and Pareto plots. No training BERT or API.

Usage: uv run python scripts/diagnose_and_plot.py --group revision-controlled --scenario fakewhatsapp-full
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

SOURCE = {
    "fakewhatsapp-full": "dataset/fakewhatsapp/train-full.csv",
    "fakewhatsapp-10pct": "dataset/fakewhatsapp/train-10pct.csv",
    "fakewhatsapp-5pct": "dataset/fakewhatsapp/train-5pct.csv",
    "fakebr-10pct": "dataset/pre-processed-train-10pct.csv",
    "fakebr-5pct": "dataset/pre-processed-train-5pct.csv",
    "covid19br-full": "dataset/covid19br/train-full.csv",
    "covid19br-10pct": "dataset/covid19br/train-10pct.csv",
    "covid19br-5pct": "dataset/covid19br/train-5pct.csv",
    "factckbr-full": "dataset/factckbr/train-full.csv",
    "factckbr-10pct": "dataset/factckbr/train-10pct.csv",
    "factckbr-5pct": "dataset/factckbr/train-5pct.csv",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, help="one output group; never mix protocols")
    ap.add_argument("--scenario", required=True)
    args = ap.parse_args()
    paths = sorted((Path("outputs") / args.group).glob("*/results.json"))
    runs = [json.loads(p.read_text()) for p in paths]
    runs = [r for r in runs if r.get("scenario", args.scenario) == args.scenario
            and "f1" in r.get("test", {})]
    if not runs:
        raise ValueError("No completed results for this group/scenario")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path("figures/revision")
    out.mkdir(parents=True, exist_ok=True)
    for key, label in (("tokens_used", "Tokens used"), ("n_synthetic", "Synthetic texts in training")):
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for arm in sorted(set(r["arm"] for r in runs)):
            subset = [r for r in runs if r["arm"] == arm]
            xs = [r.get(key, 0) for r in subset]
            ys = [r["test"]["f1"] for r in subset]
            ax.scatter(xs, ys, label=arm)
        points = sorted(((r.get(key, 0), r["test"]["f1"]) for r in runs), key=lambda p: (p[0], -p[1]))
        frontier = []
        for x, y in points:
            if not frontier or y > frontier[-1][1]:
                frontier.append((x, y))
        if len(frontier) > 1:
            ax.plot(*zip(*frontier), color="black", alpha=.5, linestyle="--", label="Pareto envelope")
        ax.set(xlabel=label, ylabel="F1 (test)", title=f"{args.scenario}: {args.group}")
        ax.legend()
        fig.tight_layout()
        name = out / f"{args.group}-{args.scenario}-f1-vs-{key}-{ts}.png"
        fig.savefig(name, dpi=300)
        plt.close(fig)
        print(name)
    # Domain classifier uses only positive real texts; exact duplicates removed.
    source = SOURCE.get(args.scenario)
    if not source or not Path(source).is_file():
        print(f"SKIP domain classifier: missing {source}")
        return
    df = pd.read_csv(source)
    real = list(dict.fromkeys(df.loc[df.label == "fake", "preprocessed_news"].dropna().astype(str)))
    for r in runs:
        synth = list(dict.fromkeys(r.get("synthetic", [])))
        # Do not train on duplicates shared by both domains.
        overlap = set(real) & set(synth)
        a, b = [x for x in real if x not in overlap], [x for x in synth if x not in overlap]
        n = min(len(a), len(b))
        if n < 10:
            continue
        a, b = a[:n], b[:n]
        x_tr, x_te, y_tr, y_te = train_test_split(a + b, [0]*n + [1]*n,
                                                    test_size=.3, random_state=42, stratify=[0]*n+[1]*n)
        clf = make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=20000),
                            LogisticRegression(max_iter=500))
        clf.fit(x_tr, y_tr)
        auc = roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1])
        print(f"real-vs-synth {r['arm']} n={n}/class AUC={auc:.3f} (random split, diagnostic only)")


if __name__ == "__main__":
    main()
