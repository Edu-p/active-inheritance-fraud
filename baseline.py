"""Braços Real (E1) e Static/brute-force (E2) do plano de experimentos."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config import Config
from src.helpers import evaluate, tune_threshold
from src.metrics import synthetic_quality
from src.registry import (
    DATASETS,
    build_embedder,
    build_generator,
    build_learner,
    load_dataset,
)
from src.runs import save_run


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Baseline arms: real | static")
    p.add_argument("--arm", choices=["real", "static"], required=True)
    p.add_argument("--csv", dest="csv_path", type=Path,
                   default=Path("dataset/pre-processed-imbalanced.csv"))
    p.add_argument("--test-csv", dest="test_csv_path", type=Path,
                   default=Path("dataset/pre-processed-test.csv"))
    p.add_argument("--val-csv", dest="val_csv_path", type=Path, default=None)
    p.add_argument("--n-samples", type=int, default=32,
                   help="fakes sintéticos (braço static)")
    p.add_argument("--generator", choices=["mock", "openai"], default="mock")
    p.add_argument("--model", dest="openai_model",
                   default=os.getenv("OPENAI_MODEL", "deepseek-flash"))
    p.add_argument("--base-url", dest="openai_base_url", default="")
    p.add_argument("--prompt-version", choices=["v1", "v2", "v3", "v4"], default="v1")
    p.add_argument("--max-run-tokens", type=int, default=1_000_000)
    p.add_argument("--gen-max-tokens", type=int, default=1024)
    p.add_argument("--generator-workers", type=int, default=1)
    p.add_argument("--zero-shot", dest="zero_shot", action="store_true")
    p.add_argument("--run-group", default="")

    p.add_argument("--learner-model", default="neuralmind/bert-base-portuguese-cased")
    p.add_argument("--learner-kind", choices=["bert", "semantic-probe"], default="bert")
    p.add_argument("--lr", dest="learner_lr", type=float, default=2e-5)
    p.add_argument("--learner-epochs", type=int, default=3)
    p.add_argument("--learner-batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", default="")
    p.add_argument("--class-weight", dest="class_weight", action="store_true")
    p.add_argument("--final-from-scratch", dest="final_from_scratch", action="store_true")
    p.add_argument("--threshold-tune", dest="threshold_tune", action="store_true")

    p.add_argument("--embedder", choices=["semantic", "tfidf"], default="semantic")
    p.add_argument("--semantic-model",
                   default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    p.add_argument("--seed", dest="random_state", type=int, default=42)
    p.add_argument("--dataset", choices=sorted(DATASETS), default="fakebr")
    p.add_argument("--dataset-variant", choices=["full", "10pct", "5pct"], default="full")
    return p.parse_args(namespace=Config())


def run_real(cfg, train, val, test, embedder):
    learner = build_learner(cfg, embedder)
    loss = learner.fit(train[0], train[1], epochs=cfg.learner_epochs)
    thr = tune_threshold(learner, val) if cfg.threshold_tune else None
    return {
        "arm": "real",
        "loss": float(loss),
        "test": evaluate(learner, test[0], test[1], thr),
        "tokens_used": 0,
        "synthetic": [],
        "n_synthetic": 0,
    }


def run_static(cfg, train, val, test, embedder, n_samples):
    generator = build_generator(cfg, train[0], train[1])

    synth = [x for x in generator.generate_batch([], n_samples) if x and x.strip()]

    learner = build_learner(cfg, embedder)
    combined_texts = train[0] + synth
    combined_y = train[1] + [1] * len(synth)
    loss = learner.fit(combined_texts, combined_y, epochs=cfg.learner_epochs)
    thr = tune_threshold(learner, val) if cfg.threshold_tune else None

    return {
        "arm": "static",
        "loss": float(loss),
        "test": evaluate(learner, test[0], test[1], thr),
        "tokens_used": int(generator.tokens_used),
        "prompt_tokens": int(getattr(generator, "prompt_tokens", 0)),
        "completion_tokens": int(getattr(generator, "completion_tokens", 0)),
        "cost_usd": float(getattr(generator, "cost_usd", 0.0)),
        "synthetic": synth,
        "n_synthetic": len(synth),
        "quality": synthetic_quality(embedder, train[0], synth),
    }


def main() -> None:
    cfg = parse_args()
    train, val, test = load_dataset(cfg)
    embedder = build_embedder(cfg, train[0])

    if cfg.arm == "real":
        out = run_real(cfg, train, val, test, embedder)
        out["quality"] = synthetic_quality(embedder, train[0], [])
    else:
        out = run_static(cfg, train, val, test, embedder, cfg.n_samples)

    print(f"\n=== {cfg.arm} — test ===")
    for k, v in out["test"].items():
        print(f"  {k:>10}: {v:.4f}")
    print(f"  tokens_used: {out['tokens_used']}")

    tag = f"static-n{cfg.n_samples}" if cfg.arm == "static" else cfg.arm
    run_dir = save_run(cfg, out, tag=tag)
    print(f"\nResults saved to {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
