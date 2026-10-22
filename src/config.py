"""Active Inheritance pipeline configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ---- Data ----
    csv_path: Path = Path("dataset/pre-processed-imbalanced.csv")  # scarce train pool
    test_csv_path: Path = Path("dataset/pre-processed-test.csv")   # balanced holdout
    val_csv_path: Path | None = None  # optional fixed/temporal validation set
    text_col: str = "preprocessed_news"
    label_col: str = "label"
    positive_label: str = "fake"  # class generated synthetically (fraudulent)

    # ---- Split ----
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42

    # ---- Embedding ----
    embedder: str = "semantic"  # "semantic" | "tfidf"
    semantic_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_features: int = 10_000

    # ---- Learner (BERT) ----
    learner_model: str = "neuralmind/bert-base-portuguese-cased"
    learner_kind: str = "bert"  # "bert" | "semantic-probe" (proxy, não JEPA/JEV)
    evaluator_kind: str = "learner"  # "learner" | "laya"
    laya_url: str = "http://127.0.0.1:8018"
    learner_lr: float = 2e-5
    learner_epochs: int = 3
    learner_batch_size: int = 16
    max_length: int = 256
    device: str = ""  # empty -> auto (cuda if available)

    # ---- Generator ----
    generator: str = "mock"  # "mock" | "openai"
    batch_size: int = 64
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""  # empty -> OpenAI default endpoint
    gen_max_tokens: int = 1024  # max_completion_tokens por amostra sintética
    prompt_version: str = "v1"  # v1/v2 historical; v3 = WhatsApp misinformation
    max_run_tokens: int = 1_000_000
    generator_workers: int = 1

    # ---- Active Inheritance loop ----
    epochs: int = 5            # T
    top_k: int = 16            # samples inherited per cycle
    w_error: float = 0.4
    w_uncertainty: float = 0.4
    w_diversity: float = 0.2

    # ---- Output ----
    output_dir: Path = Path("outputs")
    run_group: str = ""  # vazio = classificação histórica; extensões: e6/e7/e8/e9/e10/sandbox
