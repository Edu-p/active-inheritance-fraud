"""Zero-shot Laya multilingual on the unchanged Fake.br test set (NOT learner fine-tuning)."""
from __future__ import annotations

import argparse

from src.config import Config
from src.data import _texts_labels
from src.metrics import classification_metrics
from src.runs import save_run
from src.decision_scorer import LayaScorer
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8008")
    ap.add_argument("--limit", type=int, default=0, help="0 = holdout inteiro")
    args = ap.parse_args()
    cfg = Config(run_group="e9-laya", learner_kind="laya-frozen")
    texts, y = _texts_labels(pd.read_csv(cfg.test_csv_path), cfg)
    if args.limit:
        texts, y = texts[:args.limit], y[:args.limit]
    scorer = LayaScorer(args.url)
    probs = []
    for i, text in enumerate(texts):
        probs.append(float(scorer.predict([text])[0, 1]))
        if (i + 1) % 100 == 0:
            print(f"Laya: {i+1}/{len(texts)}", flush=True)
    out = {
        "arm": "laya-zero-shot",
        "checkpoint": "convaiinnovations/laya-multilingual",
        "protocol": "zero-shot; no fit; Portuguese question / Portuguese preprocessed text",
        "n_test": len(y),
        "test": classification_metrics(y, [int(p >= 0.5) for p in probs]),
        "prob_fake": probs,
        "labels": y,
        "tokens_used": 0,
    }
    run_dir = save_run(cfg, out, tag="laya-zero-shot")
    print(f"Laya zero-shot F1={out['test']['f1']:.4f}; saved {run_dir}")


if __name__ == "__main__":
    main()
