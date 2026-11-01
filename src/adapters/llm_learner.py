"""Jev-style decision model: frozen causal-LLM backbone + LoRA + classification head.

Imitates the open "Kev" reconstruction of the hosted Jev: a frozen Qwen-style
backbone with a low-rank adapter and a pointer/classification head that returns
P(y|x) WITHOUT text generation. Multilingual backbone (Qwen2.5) so it works on
PT-BR (unlike Kev, which is English-only).

Exposes the same Learner API (fit/predict/compute_uncertainty/error/reset) so the
warm-start replay loop uses it unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class LoraLLMLearner:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        lr: float = 2e-4,
        batch_size: int = 8,
        max_length: int = 256,
        device: str | None = None,
        random_state: int = 42,
        class_weight: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ) -> None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.model_name = model_name
        self.random_state = random_state
        self.lr = lr
        self.class_weight = class_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self._build()

    def _build(self):
        base = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2, dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        if getattr(base.config, "pad_token_id", None) is None:
            base.config.pad_token_id = self.tokenizer.pad_token_id
        lora = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(base, lora)
        model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=self.lr
        )
        return model

    def reset(self) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        del self.model
        torch.cuda.empty_cache()
        self.model = self._build()

    @staticmethod
    def _class_weights(y: list[int]) -> torch.Tensor:
        counts = {0: y.count(0), 1: y.count(1)}
        n = len(y)
        return torch.tensor(
            [n / (2.0 * counts[0]) if counts[0] else 1.0,
             n / (2.0 * counts[1]) if counts[1] else 1.0],
            dtype=torch.float,
        )

    def _encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def train_step(self, texts: list[str], y: list[int], sample_weight=None) -> float:
        self.model.train()
        labels = torch.tensor(y, dtype=torch.long, device=self.device)
        if sample_weight is not None:
            sample_weight = sample_weight.to(self.device)
        out = self.model(**self._encode(texts))
        loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=sample_weight)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def predict(self, texts: list[str]) -> np.ndarray:
        self.model.eval()
        probs = []
        for i in range(0, len(texts), self.batch_size):
            out = self.model(**self._encode(texts[i : i + self.batch_size]))
            probs.append(torch.softmax(out.logits, dim=-1).float().cpu().numpy())
        return np.concatenate(probs) if probs else np.empty((0, 2))

    def predict_class(self, texts, threshold=None):
        p = self.predict(texts)
        return p.argmax(axis=1) if threshold is None else (p[:, 1] >= threshold).astype(int)

    def compute_uncertainty(self, texts):
        p = np.clip(self.predict(texts), 1e-12, 1.0)
        return -(p * np.log(p)).sum(axis=1) / np.log(2.0)

    def error(self, texts, y):
        p = self.predict(texts)
        return -np.log(np.clip(p[np.arange(len(y)), np.asarray(y, dtype=int)], 1e-12, 1.0))

    def fit(self, texts, y, epochs=3, from_scratch=False):
        if from_scratch:
            self.reset()
        weights = self._class_weights(y) if self.class_weight else None
        order = np.random.permutation(len(texts))
        losses = []
        for _ in range(epochs):
            for i in range(0, len(order), self.batch_size):
                b = order[i : i + self.batch_size]
                losses.append(self.train_step([texts[j] for j in b], [y[j] for j in b], weights))
        return float(np.mean(losses)) if losses else 0.0
