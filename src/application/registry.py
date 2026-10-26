"""Composition root: registries + factories que montam os adapters a partir da Config.

O wiring hexagonal vive aqui. Para adicionar uma feature (gerador, learner,
embedder, scorer ou dataset novos), registre UMA função aqui e a CLI já a
pega pela string da Config (--generator, --learner-kind, --embedder,
--evaluator, --dataset). Sem editar main.py/baseline.py.

Registre com:
    from src.registry import register_generator
    @register_generator("meu-novo")
    def _meu(cfg, train_texts, train_y):
        return MeuGerador(cfg)
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..adapters.data import load_experiment_data
from ..adapters.decision_scorer import LayaScorer
from ..adapters.embedder import SemanticEmbedder, TfidfEmbedder
from ..domain.evaluator import AdversarialEvaluator
from ..adapters.generator import MockGenerator, OpenAIGenerator
from ..adapters.learner import Learner
from ..adapters.probe import SemanticProbe

GENERATORS: dict = {}
LEARNERS: dict = {}
EMBEDDERS: dict = {}
SCORERS: dict = {}
DATASETS: dict = {}


def register_generator(name):
    def deco(fn):
        GENERATORS[name] = fn
        return fn
    return deco


def register_learner(name):
    def deco(fn):
        LEARNERS[name] = fn
        return fn
    return deco


def register_embedder(name):
    def deco(fn):
        EMBEDDERS[name] = fn
        return fn
    return deco


def register_scorer(name):
    def deco(fn):
        SCORERS[name] = fn
        return fn
    return deco


def register_dataset(name):
    def deco(fn):
        DATASETS[name] = fn
        return fn
    return deco


# ---- Generators (cfg, train_texts, train_y) -> GeneratorPort ----
@register_generator("mock")
def _mock_generator(cfg, train_texts, train_y):
    seed_fake = [t for t, lab in zip(train_texts, train_y) if lab == 1]
    return MockGenerator(seed_fake, random_state=cfg.random_state)


@register_generator("openai")
def _openai_generator(cfg, *_):
    return OpenAIGenerator(cfg)


# ---- Learners (cfg, embedder) -> LearnerPort ----
@register_learner("bert")
def _bert_learner(cfg, embedder=None):
    return Learner(
        model_name=cfg.learner_model,
        lr=cfg.learner_lr,
        batch_size=cfg.learner_batch_size,
        max_length=cfg.max_length,
        device=cfg.device or None,
        random_state=cfg.random_state,
        class_weight=cfg.class_weight,
    )


@register_learner("semantic-probe")
def _probe_learner(cfg, embedder=None):
    if cfg.embedder != "semantic" or embedder is None:
        raise ValueError("semantic-probe exige --embedder semantic")
    return SemanticProbe(embedder)


# ---- Embedders (cfg) -> EmbedderPort ----
@register_embedder("tfidf")
def _tfidf_embedder(cfg):
    return TfidfEmbedder(max_features=cfg.max_features)


@register_embedder("semantic")
def _semantic_embedder(cfg):
    return SemanticEmbedder(cfg.semantic_model)


# ---- Scorers (cfg, learner, embedder) -> ScorerPort ----
@register_scorer("learner")
def _learner_scorer(cfg, learner, embedder):
    return learner


@register_scorer("laya")
def _laya_scorer(cfg, learner, embedder):
    return LayaScorer(cfg.laya_url)


# ---- Datasets (cfg) -> (train, val, test) ----
# Os novos datasets são preparados por scripts/prepare_*.py e seguem o mesmo
# schema (label, preprocessed_news). Basta registrar o nome + arquivos aqui.
_VARIANT_TRAIN = {
    "full": "train-full.csv",
    "10pct": "train-10pct.csv",
    "5pct": "train-5pct.csv",
}
_DATASET_FILES = {
    "covid19br": ("test-june.csv", "validation-june.csv"),
    "factckbr": ("test.csv", "validation.csv"),
    "fakewhatsapp": ("test-october.csv", "validation-september.csv"),
}


@register_dataset("fakebr")
def _fakebr(cfg):
    return load_experiment_data(cfg)


@register_dataset("covid19br")
@register_dataset("factckbr")
@register_dataset("fakewhatsapp")
def _prepared_dataset(cfg):
    v = getattr(cfg, "dataset_variant", "full")
    test, val = _DATASET_FILES[cfg.dataset]
    base = Path("dataset") / cfg.dataset
    return load_experiment_data(
        replace(
            cfg,
            csv_path=base / _VARIANT_TRAIN[v],
            test_csv_path=base / test,
            val_csv_path=base / val,
        )
    )


# ---- Composition root ----
def build_embedder(cfg, texts):
    return EMBEDDERS[cfg.embedder](cfg).fit(texts)


def build_learner(cfg, embedder=None):
    return LEARNERS[cfg.learner_kind](cfg, embedder)


def build_generator(cfg, train_texts, train_y):
    return GENERATORS[cfg.generator](cfg, train_texts, train_y)


def build_scorer(cfg, learner, embedder):
    return SCORERS[cfg.evaluator_kind](cfg, learner, embedder)


def build_evaluator(cfg, learner, embedder):
    scorer = build_scorer(cfg, learner, embedder)
    return AdversarialEvaluator(
        scorer,
        embedder,
        w_error=cfg.w_error,
        w_uncertainty=cfg.w_uncertainty,
        w_diversity=cfg.w_diversity,
    )


def load_dataset(cfg):
    return DATASETS[cfg.dataset](cfg)
