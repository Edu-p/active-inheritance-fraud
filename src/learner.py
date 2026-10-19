"""Downstream classifier (the "learner") — small BERT via PyTorch/HF.

Exposes the required API: train_step (one backprop step), predict (softmax)
and compute_uncertainty (softmax entropy).
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Learner:
    def __init__(
        self,
        model_name: str = "prajjwal1/bert-tiny",
        lr: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 256,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def _encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def train_step(self, texts: list[str], y: list[int]) -> float:
        """One backprop step over a batch; returns the loss."""
        self.model.train()
        labels = torch.tensor(y, dtype=torch.long, device=self.device)
        out = self.model(**self._encode(texts), labels=labels)
        self.optimizer.zero_grad()
        out.loss.backward()
        self.optimizer.step()
        return float(out.loss.item())

    @torch.no_grad()
    def predict(self, texts: list[str]) -> np.ndarray:
        """Softmax P(y|x) → (n, 2) matrix."""
        self.model.eval()
        probs: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            logits = self.model(**self._encode(texts[i : i + self.batch_size])).logits
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probs) if probs else np.empty((0, 2))

    def predict_class(self, texts: list[str]) -> np.ndarray:
        return self.predict(texts).argmax(axis=1)

    def compute_uncertainty(self, texts: list[str]) -> np.ndarray:
        """Normalized softmax entropy → [0, 1]."""
        p = np.clip(self.predict(texts), 1e-12, 1.0)
        return -(p * np.log(p)).sum(axis=1) / np.log(2.0)

    def error(self, texts: list[str], y: list[int]) -> np.ndarray:
        """Per-sample log-loss: -log P(true class)."""
        p = self.predict(texts)
        p_true = p[np.arange(len(y)), np.asarray(y, dtype=int)]
        return -np.log(np.clip(p_true, 1e-12, 1.0))

    def fit(self, texts: list[str], y: list[int], epochs: int = 3) -> float:
        """Warm-start fine-tune over real + inherited; returns the mean loss."""
        order = np.random.permutation(len(texts))
        losses: list[float] = []
        for _ in range(epochs):
            for i in range(0, len(order), self.batch_size):
                b = order[i : i + self.batch_size]
                losses.append(
                    self.train_step([texts[j] for j in b], [y[j] for j in b])
                )
        return float(np.mean(losses)) if losses else 0.0
