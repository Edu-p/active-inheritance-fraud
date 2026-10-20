"""Self-check of the core (no torch): multi-objective scoring + mock generator + metrics.

Run with: python tests/test_core.py
"""
import numpy as np

from src.embedder import TfidfEmbedder
from src.generator import MockGenerator
from src.evaluator import AdversarialEvaluator
from src.metrics import classification_metrics, diversity_metrics
from src.runs import experiment_tag, save_run
from src.probe import SemanticProbe
from src.config import Config
from src.generator import OpenAIGenerator
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from src.decision_scorer import LayaScorer


class StubLearner:
    """Constant error, increasing uncertainty → the score should pick the last sample."""

    def error(self, texts, y):
        return np.full(len(texts), 0.5)

    def compute_uncertainty(self, texts):
        return np.linspace(0.0, 1.0, len(texts))


def main() -> None:
    seeds = [f"fake news number {i} for testing" for i in range(10)]

    gen = MockGenerator(seeds, random_state=0)
    batch = gen.generate_batch([], 8)
    assert len(batch) == 8 and all(t and isinstance(t, str) for t in batch)

    emb = TfidfEmbedder(max_features=500).fit(seeds)
    ev = AdversarialEvaluator(StubLearner(), emb)
    selected, mean_score, parts = ev.select(batch, [1] * 8, [], 4)
    assert len(selected) == 4 and selected[0] == batch[-1]  # highest uncertainty
    assert 0.0 <= mean_score <= 1.0
    assert set(parts) == {"error", "uncertainty", "diversity"}

    m = classification_metrics([0, 1, 1, 0], [0, 1, 0, 1])
    assert m["f1"] == 0.5 and set(m) == {"accuracy", "f1", "precision", "recall"}

    d = diversity_metrics(seeds, emb)
    assert 0.0 <= d["mean_pairwise_sim"] <= 1.0

    # mapeamento experimento -> subdiretório
    class Cfg:
        top_k = 16
        w_error = 0.4
        w_uncertainty = 0.4
        w_diversity = 0.2
    assert experiment_tag(Cfg(), "real") == "e1"
    assert experiment_tag(Cfg(), "static") == "e2"
    assert experiment_tag(Cfg(), "ai") == "e3"

    class Weights(Cfg):
        w_error = 0.7
        w_uncertainty = 0.2
        w_diversity = 0.1
    assert experiment_tag(Weights(), "ai") == "e4"

    class TopK(Cfg):
        top_k = 8
    assert experiment_tag(TopK(), "ai") == "e5"
    assert experiment_tag(Cfg(), "weird") == "sandbox"

    # Dois saves consecutivos não podem sobrescrever um resultado.
    with TemporaryDirectory() as tmp:
        cfg = Config(output_dir=Path(tmp), run_group="e6-5pct")
        a = save_run(cfg, {"arm": "real", "test": {"f1": 0.1}}, "real")
        b = save_run(cfg, {"arm": "real", "test": {"f1": 0.2}}, "real")
        assert a != b and (a / "results.json").exists() and (b / "results.json").exists()

    class StubEmbedder:
        def transform(self, texts):
            return np.asarray([[float("bom" in t), float("ruim" in t)] for t in texts])

    probe = SemanticProbe(StubEmbedder())
    probe.fit(["bom", "bom bom", "ruim", "ruim ruim"], [1, 1, 0, 0])
    assert probe.predict_class(["bom", "ruim"]).tolist() == [1, 0]
    assert len(probe.compute_uncertainty(["bom"])) == 1
    assert len(probe.error(["ruim"], [0])) == 1

    gen = OpenAIGenerator.__new__(OpenAIGenerator)
    gen.prompt_version, gen.calls = "v2", 0
    system, user = gen._prompt([])
    assert "FICTÍCIO" in system and "saúde pública" in user

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"answers":{"veracity":{"probabilities":{"fake":0.8,"true":0.2}}}}'
    with patch("src.decision_scorer.urlopen", return_value=Response()) as request:
        scorer = LayaScorer("http://127.0.0.1:8018")
        assert scorer.predict(["texto"])[0, 1] == 0.8
        assert len(scorer.error(["texto"], [1])) == 1
        assert len(scorer.compute_uncertainty(["texto"])) == 1
        assert request.call_count == 1  # cache: score erro + incerteza sem 2ª chamada
        body = __import__("json").loads(request.call_args.args[0].data)
        assert body["state"] == "texto" and body["questions"]["veracity"]["type"] == "choice"

    print("OK — core (scoring, mock generator, metrics) passed.")


if __name__ == "__main__":
    main()
