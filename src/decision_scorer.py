"""Laya multilingual frozen decision scorer over the local /v1/systemone endpoint."""
from __future__ import annotations

import json
from urllib.request import Request, urlopen

import numpy as np


class LayaScorer:
    """Judge synthetic fake-news difficulty; BERT remains the downstream learner."""

    def __init__(self, url="http://127.0.0.1:8018"):
        self.url = url.rstrip("/") + "/v1/systemone"
        self.cache = {}

    def _probability(self, text):
        payload = {
            "model": "laya-multilingual",
            "state": text,
            "questions": {
                "veracity": {
                    "type": "choice",
                    "instructions": "Este texto em português é uma notícia falsa ou verdadeira?", 
                    "criteria": {
                        "fake": "Notícia inventada ou afirmação falsa",
                        "true": "Notícia real ou relato factual", 
                    },
                },
            },
        }
        req = Request(self.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=120) as response:
            data = json.load(response)
        p = float(data["answers"]["veracity"]["probabilities"]["fake"])
        if not 0 <= p <= 1:
            raise ValueError("Laya retornou probabilidade inválida")
        return p

    def predict(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        for text in missing:
            self.cache[text] = self._probability(text)
        ps = np.asarray([self.cache[t] for t in texts])
        return np.column_stack((1 - ps, ps))

    def compute_uncertainty(self, texts):
        p = np.clip(self.predict(texts), 1e-12, 1)
        return -(p * np.log(p)).sum(axis=1) / np.log(2)

    def error(self, texts, y):
        p = self.predict(texts)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
