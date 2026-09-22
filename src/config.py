"""Configuração do pipeline de Herança Ativa."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ---- Dados ----
    csv_path: Path = Path("dataset/pre-processed.csv")
    text_col: str = "preprocessed_news"
    label_col: str = "label"
    positive_label: str = "fake"  # classe gerada sinteticamente (fraudulenta)

    # ---- Split ----
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42

    # ---- Embedding ----
    max_features: int = 10_000

    # ---- Learner (BERT) ----
    learner_model: str = "neuralmind/bert-base-portuguese-cased"
    learner_lr: float = 2e-5
    learner_epochs: int = 3
    learner_batch_size: int = 16
    max_length: int = 256
    device: str = ""  # vazio -> auto (cuda se disponível)

    # ---- Gerador ----
    generator: str = "mock"  # "mock" | "openai"
    batch_size: int = 64
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""  # vazio -> endpoint padrão da OpenAI

    # ---- Loop de Herança Ativa ----
    epochs: int = 5            # T
    top_k: int = 16            # amostras herdadas por ciclo
    w_error: float = 0.4
    w_uncertainty: float = 0.4
    w_diversity: float = 0.2

    # ---- Saída ----
    output_dir: Path = Path("outputs")
