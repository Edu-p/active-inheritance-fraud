"""Self-check of the core (no torch): multi-objective scoring + mock generator + metrics.

Run with: python tests/test_core.py
"""
import numpy as np

from src.embedder import TfidfEmbedder
from src.generator import MockGenerator
from src.evaluator import AdversarialEvaluator
from src.metrics import (
    best_threshold,
    classification_metrics,
    diversity_metrics,
    probabilistic_metrics,
)
from src.runs import experiment_tag, save_run
from src.probe import SemanticProbe
from src.config import Config
from src.generator import OpenAIGenerator
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from src.decision_scorer import LayaScorer
from src.text import normalize_fakebr
from scripts.controlled_curatorship import select_indices
from scripts.revision_status import completed_replays
from src.learner import Learner
import random


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

    pm = probabilistic_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert set(pm) == {"roc_auc", "pr_auc"} and 0.0 <= pm["roc_auc"] <= 1.0
    thr = best_threshold([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert 0.05 <= thr <= 0.95

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
        cfg = Config(output_dir=Path(tmp), run_group="replay-smoke")
        a = save_run(cfg, {"arm": "real", "test": {"f1": 0.1}}, "real")
        b = save_run(cfg, {"arm": "real", "test": {"f1": 0.2}}, "real")
        assert a != b and (a / "results.json").exists() and (b / "results.json").exists()

    # grupo histórico não pode ser sobrescrito
    try:
        save_run(Config(output_dir=Path(tmp), run_group="e1"), {"arm": "real"}, "real")
        raise AssertionError("deveria ter protegido grupo histórico e1")
    except ValueError:
        pass

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
    gen.prompt_version = "v3"
    system, user = gen._prompt([], 1)
    assert "WhatsApp" in system and "eleições" in user and "Markdown" in system
    gen.zero_shot = True
    system, user = gen._prompt(["exemplo herdado que não deve aparecer"])
    assert "exemplo herdado" not in user
    gen.prompt_version = "v4"
    system, user = gen._prompt([], 3)
    assert "201 a 350" in user and "URGENTE" in system
    assert normalize_fakebr("Á notícia, e o FATO!") == "noticia fato"

    class Scores:
        def score(self, candidates, labels, inherited):
            return np.asarray([0.1, 0.8, 0.4]), {}
    items = ["a", "b", "c"]
    assert select_indices("all", items, 1, random.Random(42)) == [0, 1, 2]
    assert select_indices("easy-k", items, 1, random.Random(42), Scores()) == [0]
    assert select_indices("score-k", items, 1, random.Random(42), Scores()) == [1]
    assert select_indices("random-k", items, 2, random.Random(42)) == [0, 2]
    weights = Learner._class_weights([0, 0, 0, 1]).tolist()
    assert np.allclose(weights, [4/6, 2.0])
    with TemporaryDirectory() as tmp:
        result = Path(tmp) / "progress" / "run" / "results.json"
        result.parent.mkdir(parents=True)
        result.write_text('{"arm":"random-k","test":{"f1":0.7},"config":{"random_state":"42"}}')
        assert (42, "random-k") in completed_replays(Path(tmp), "progress", ("random-k",))

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"answers":{"veracity":{"probabilities":{"fake":0.8,"true":0.2}}}}'
    with patch("src.adapters.decision_scorer.urlopen", return_value=Response()) as request:
        scorer = LayaScorer("http://127.0.0.1:8018")
        assert scorer.predict(["texto"])[0, 1] == 0.8
        assert len(scorer.error(["texto"], [1])) == 1
        assert len(scorer.compute_uncertainty(["texto"])) == 1
        assert request.call_count == 1  # cache: score erro + incerteza sem 2ª chamada
        body = __import__("json").loads(request.call_args.args[0].data)
        assert body["state"] == "texto" and body["questions"]["veracity"]["type"] == "choice"

    # ---- arquitetura: composition root + ports + helpers + report ----
    from src.registry import (
        DATASETS, EMBEDDERS, GENERATORS, LEARNERS, SCORERS,
        _VARIANT_TRAIN, _DATASET_FILES,
    )
    assert set(DATASETS) == {"fakebr", "covid19br", "factckbr", "fakewhatsapp"}
    assert set(GENERATORS) == {"mock", "openai"}
    assert set(LEARNERS) == {"bert", "semantic-probe"}
    assert set(EMBEDDERS) == {"semantic", "tfidf"}
    assert set(SCORERS) == {"learner", "laya"}
    assert _VARIANT_TRAIN["10pct"] == "train-10pct.csv"
    assert set(_DATASET_FILES) == {"covid19br", "factckbr", "fakewhatsapp"}

    from src.helpers import evaluate, tune_threshold

    class StubProbe:
        def predict(self, texts):
            return np.asarray([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.8, 0.2]])

    ev = evaluate(StubProbe(), ["a"] * 4, [0, 1, 1, 0])
    assert set(ev) >= {"f1", "roc_auc", "threshold"} and ev["threshold"] == 0.5
    thr = tune_threshold(StubProbe(), (["a"] * 4, [0, 1, 1, 0]))
    assert 0.05 <= thr <= 0.95

    from src.report import latex_table, mean_sd, fmt, esc
    m, s = mean_sd([0.8, 0.9, 0.85])
    assert abs(m - 0.85) < 1e-9 and s >= 0
    t = latex_table(["A", "B"], [["x", fmt((m, s))], ["y", "--"]], caption="C", label="tab:x")
    assert "\\toprule" in t and "\\caption{C}" in t and "\\label{tab:x}" in t and "x & " in t
    assert esc("a_b%c") == "a\\_b\\%c"

    print("OK — core (scoring, mock generator, metrics) passed.")


if __name__ == "__main__":
    main()
