"""Braços Real (E1) e Static/brute-force (E2) do plano de experimentos."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config import Config
from src.data import load_experiment_data
from src.embedder import SemanticEmbedder, TfidfEmbedder
from src.generator import MockGenerator, OpenAIGenerator
from src.learner import Learner
from src.probe import SemanticProbe
from src.metrics import classification_metrics, synthetic_quality
from src.runs import save_run


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Baseline arms: real | static")
    p.add_argument("--arm", choices=["real", "static"], required=True)
    p.add_argument("--csv", dest="csv_path", type=Path,
                   default=Path("dataset/pre-processed-imbalanced.csv"))
    p.add_argument("--test-csv", dest="test_csv_path", type=Path,
                   default=Path("dataset/pre-processed-test.csv"))
    p.add_argument("--n-samples", type=int, default=32,
                   help="fakes sintéticos (braço static)")
    p.add_argument("--generator", choices=["mock", "openai"], default="mock")
    p.add_argument("--model", dest="openai_model",
                   default=os.getenv("OPENAI_MODEL", "deepseek-flash"))
    p.add_argument("--base-url", dest="openai_base_url", default="")
    p.add_argument("--prompt-version", choices=["v1", "v2"], default="v1")
    p.add_argument("--max-run-tokens", type=int, default=1_000_000)
    p.add_argument("--run-group", choices=["e6-5pct", "e7-probe", "e8-generators", "sandbox"], default="")

    p.add_argument("--learner-model", default="neuralmind/bert-base-portuguese-cased")
    p.add_argument("--learner-kind", choices=["bert", "semantic-probe"], default="bert")
    p.add_argument("--lr", dest="learner_lr", type=float, default=2e-5)
    p.add_argument("--learner-epochs", type=int, default=3)
    p.add_argument("--learner-batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", default="")

    p.add_argument("--embedder", choices=["semantic", "tfidf"], default="semantic")
    p.add_argument("--semantic-model",
                   default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    p.add_argument("--seed", dest="random_state", type=int, default=42)
    return p.parse_args(namespace=Config())


def build_embedder(cfg, texts):
    if cfg.embedder == "tfidf":
        return TfidfEmbedder(max_features=cfg.max_features).fit(texts)
    return SemanticEmbedder(cfg.semantic_model).fit(texts)


def new_learner(cfg, embedder=None):
    if cfg.learner_kind == "semantic-probe":
        if cfg.embedder != "semantic" or embedder is None:
            raise ValueError("semantic-probe exige --embedder semantic")
        return SemanticProbe(embedder)
    return Learner(
        model_name=cfg.learner_model,
        lr=cfg.learner_lr,
        batch_size=cfg.learner_batch_size,
        max_length=cfg.max_length,
        device=cfg.device or None,
        random_state=cfg.random_state,
    )


def evaluate(learner, test):
    return classification_metrics(test[1], learner.predict_class(test[0]))


def run_real(cfg, train, test, embedder):
    learner = new_learner(cfg, embedder)
    loss = learner.fit(train[0], train[1], epochs=cfg.learner_epochs)
    return {
        "arm": "real",
        "loss": float(loss),
        "test": evaluate(learner, test),
        "tokens_used": 0,
        "synthetic": [],
        "n_synthetic": 0,
    }


def run_static(cfg, train, test, embedder, n_samples):
    if cfg.generator == "mock":
        seed_fake = [t for t, lab in zip(train[0], train[1]) if lab == 1]
        generator = MockGenerator(seed_fake, random_state=cfg.random_state)
    else:
        generator = OpenAIGenerator(cfg)

    synth = [x for x in generator.generate_batch([], n_samples) if x and x.strip()]

    learner = new_learner(cfg, embedder)
    combined_texts = train[0] + synth
    combined_y = train[1] + [1] * len(synth)
    loss = learner.fit(combined_texts, combined_y, epochs=cfg.learner_epochs)

    return {
        "arm": "static",
        "loss": float(loss),
        "test": evaluate(learner, test),
        "tokens_used": int(generator.tokens_used),
        "synthetic": synth,
        "n_synthetic": len(synth),
        "quality": synthetic_quality(embedder, train[0], synth),
    }


def main() -> None:
    cfg = parse_args()
    train, _, test = load_experiment_data(cfg)
    embedder = build_embedder(cfg, train[0])

    if cfg.arm == "real":
        out = run_real(cfg, train, test, embedder)
        out["quality"] = synthetic_quality(embedder, train[0], [])
    else:
        out = run_static(cfg, train, test, embedder, cfg.n_samples)

    print(f"\n=== {cfg.arm} — test ===")
    for k, v in out["test"].items():
        print(f"  {k:>10}: {v:.4f}")
    print(f"  tokens_used: {out['tokens_used']}")

    tag = f"static-n{cfg.n_samples}" if cfg.arm == "static" else cfg.arm
    run_dir = save_run(cfg, out, tag=tag)
    print(f"\nResults saved to {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
