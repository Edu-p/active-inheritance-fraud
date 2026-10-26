"""Ports hexagonais: contratos dos quais o núcleo depende.

Os adapters em `src/` já implementam estes contratos por duck typing. Aqui
eles ficam explícitos para que um adapter novo (dataset, gerador, learner,
embedder ou scorer) seja adicionado sem tocar no núcleo (pipeline/evaluator).

ponytail: Protocols são só documentação + tipagem estática; nada registra
contra eles em runtime. O wiring real vive em `src/registry.py`.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class LearnerPort(Protocol):
    """Classifier downstream: treina e expõe erro/incerteza para o evaluator."""

    def fit(self, texts, y, epochs=..., from_scratch=...) -> float: ...
    def predict(self, texts) -> np.ndarray: ...            # (n, 2) softmax
    def predict_class(self, texts, threshold=...) -> np.ndarray: ...
    def compute_uncertainty(self, texts) -> np.ndarray: ...  # entropia [0,1]
    def error(self, texts, y) -> np.ndarray: ...             # -log P(classe real)


class GeneratorPort(Protocol):
    """Gerador de texto sintético."""

    def generate_batch(self, inherited_texts, n) -> list[str]: ...


class EmbedderPort(Protocol):
    """Embedder de texto (fit é no-op para modelos frozen)."""

    def fit(self, texts): ...
    def transform(self, texts): ...


class ScorerPort(Protocol):
    """Backend do evaluator adversarial (learner ou juiz externo como Laya)."""

    def predict(self, texts) -> np.ndarray: ...
    def compute_uncertainty(self, texts) -> np.ndarray: ...
    def error(self, texts, y) -> np.ndarray: ...
