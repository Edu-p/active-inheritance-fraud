"""CLI entry point for the Active Inheritance pipeline."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config import Config
from src.data import load_experiment_data
from src.embedder import SemanticEmbedder, TfidfEmbedder
from src.learner import Learner
from src.generator import MockGenerator, OpenAIGenerator
from src.evaluator import AdversarialEvaluator
from src.metrics import synthetic_quality
from src.pipeline import ActiveInheritancePipeline
from src.runs import save_run


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Active-Inheritance Guided Synthetic Generation"
    )
    p.add_argument("--csv", dest="csv_path", type=Path,
                   default=Path("dataset/pre-processed-imbalanced.csv"))
    p.add_argument("--test-csv", dest="test_csv_path", type=Path,
                   default=Path("dataset/pre-processed-test.csv"))
    p.add_argument("--generator", choices=["mock", "openai"], default="mock")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--w-error", dest="w_error", type=float, default=0.4)
    p.add_argument("--w-uncertainty", dest="w_uncertainty", type=float, default=0.4)
    p.add_argument("--w-diversity", dest="w_diversity", type=float, default=0.2)

    # LLM (generator)
    p.add_argument("--model", dest="openai_model",
                   default=os.getenv("OPENAI_MODEL", "deepseek-flash"))
    p.add_argument("--base-url", dest="openai_base_url", default="")

    # Learner (BERT)
    p.add_argument("--learner-model", default="neuralmind/bert-base-portuguese-cased")
    p.add_argument("--lr", dest="learner_lr", type=float, default=2e-5)
    p.add_argument("--learner-epochs", type=int, default=3)
    p.add_argument("--learner-batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", default="")

    # Embeddings
    p.add_argument("--embedder", choices=["semantic", "tfidf"], default="semantic")
    p.add_argument("--semantic-model",
                   default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    p.add_argument("--seed", dest="random_state", type=int, default=42)
    return p.parse_args(namespace=Config())


def build_embedder(cfg, texts):
    if cfg.embedder == "tfidf":
        return TfidfEmbedder(max_features=cfg.max_features).fit(texts)
    return SemanticEmbedder(cfg.semantic_model).fit(texts)


def main() -> None:
    cfg = parse_args()
    train, val, test = load_experiment_data(cfg)
    embedder = build_embedder(cfg, train[0])

    learner = Learner(
        model_name=cfg.learner_model,
        lr=cfg.learner_lr,
        batch_size=cfg.learner_batch_size,
        max_length=cfg.max_length,
        device=cfg.device or None,
        random_state=cfg.random_state,
    )

    if cfg.generator == "mock":
        seed_fake = [t for t, lab in zip(train[0], train[1]) if lab == 1]
        generator = MockGenerator(seed_fake, random_state=cfg.random_state)
    else:
        generator = OpenAIGenerator(cfg)

    evaluator = AdversarialEvaluator(
        learner, embedder,
        w_error=cfg.w_error,
        w_uncertainty=cfg.w_uncertainty,
        w_diversity=cfg.w_diversity,
    )

    pipeline = ActiveInheritancePipeline(cfg, learner, embedder, generator, evaluator)
    history, test_metrics, inherited = pipeline.run(train, val, test)

    print("\n=== Active Inheritance — per epoch ===")
    print(f"{'epoch':>5} {'loss':>8} {'val_f1':>8} {'val_prec':>9} {'val_rec':>9} "
          f"{'pool_n':>7} {'pool_sim':>9} {'score':>7} {'tokens':>8}")
    for h in history:
        print(
            f"{h['epoch']:>5} {h['loss']:>8.4f} {h['val']['f1']:>8.4f} "
            f"{h['val']['precision']:>9.4f} {h['val']['recall']:>9.4f} "
            f"{h['pool_diversity']['n']:>7} "
            f"{h['pool_diversity']['mean_pairwise_sim']:>9.4f} "
            f"{h['mean_score_topk']:>7.3f} {h['tokens_used']:>8}"
        )

    print("\n=== Downstream test ===")
    for k, v in test_metrics.items():
        print(f"  {k:>10}: {v:.4f}")
    total_tokens = history[-1]["tokens_used"] if history else 0
    print(f"  total_tokens: {total_tokens}")

    out = {
        "arm": "ai",
        "history": history,
        "test": test_metrics,
        "tokens_used": int(total_tokens),
        "synthetic": inherited,
        "n_synthetic": len(inherited),
        "quality": synthetic_quality(embedder, train[0], inherited),
    }
    run_dir = save_run(cfg, out, tag="ai")
    print(f"\nResults saved to {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
