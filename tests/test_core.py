"""Self-check do núcleo (sem torch): scoring multiobjetivo + mock generator + métricas.

Roda com: python tests/test_core.py
"""
import numpy as np

from src.embedder import TfidfEmbedder
from src.generator import MockGenerator
from src.evaluator import AdversarialEvaluator
from src.metrics import classification_metrics, diversity_metrics


class StubLearner:
    """Erro constante, incerteza crescente → o score deve eleger a última amostra."""

    def error(self, texts, y):
        return np.full(len(texts), 0.5)

    def compute_uncertainty(self, texts):
        return np.linspace(0.0, 1.0, len(texts))


def main() -> None:
    seeds = [f"notícia fake número {i} para teste" for i in range(10)]

    gen = MockGenerator(seeds, random_state=0)
    batch = gen.generate_batch([], 8)
    assert len(batch) == 8 and all(t and isinstance(t, str) for t in batch)

    emb = TfidfEmbedder(max_features=500).fit(seeds)
    ev = AdversarialEvaluator(StubLearner(), emb)
    selected, mean_score, parts = ev.select(batch, [1] * 8, [], 4)
    assert len(selected) == 4 and selected[0] == batch[-1]  # maior incerteza
    assert 0.0 <= mean_score <= 1.0
    assert set(parts) == {"error", "uncertainty", "diversity"}

    m = classification_metrics([0, 1, 1, 0], [0, 1, 0, 1])
    assert m["f1"] == 0.5 and set(m) == {"accuracy", "f1", "precision", "recall"}

    d = diversity_metrics(seeds, emb)
    assert 0.0 <= d["mean_pairwise_sim"] <= 1.0

    print("OK — núcleo (scoring, gerador mock, métricas) passou.")


if __name__ == "__main__":
    main()
