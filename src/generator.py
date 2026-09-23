"""Synthetic text generators.

- MockGenerator: perturbs real fake news, so the pipeline runs offline.
- OpenAIGenerator: uses an LLM (OpenAI-compatible API).
"""
from __future__ import annotations

import os
import random

from dotenv import load_dotenv


class BaseGenerator:
    def __init__(self):
        self.tokens_used = 0

    def generate_batch(self, inherited_texts, n):
        raise NotImplementedError


class MockGenerator(BaseGenerator):
    """Simulates synthetic generation by perturbing real fake news."""

    def __init__(self, seed_fake_texts, random_state: int = 42):
        super().__init__()
        self.seed = seed_fake_texts
        self.rng = random.Random(random_state)

    def _perturb(self, text):
        words = text.split()
        if len(words) < 6:
            return text
        out = [w for w in words if self.rng.random() > 0.15]  # dropout
        if not out:
            out = words[:5]
        if len(out) >= 4:
            i = self.rng.randrange(len(out) - 3)
            window = out[i:i + 4]
            self.rng.shuffle(window)
            out[i:i + 4] = window
        return " ".join(out)

    def generate_batch(self, inherited_texts, n):
        pool = inherited_texts if inherited_texts else self.seed
        bases = self.rng.choices(pool, k=n)
        batch = [self._perturb(t) for t in bases]
        self.tokens_used += sum(len(t.split()) for t in batch)
        return batch


class OpenAIGenerator(BaseGenerator):
    def __init__(self, cfg):
        super().__init__()
        load_dotenv()
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        base_url = cfg.openai_base_url or os.getenv("OPENAI_BASE_URL") or None
        model = cfg.openai_model or os.getenv("OPENAI_MODEL")

        if not api_key:
            raise ValueError(
                "Set OPENAI_API_KEY (or DEEPSEEK_API_KEY) in the environment or .env"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _prompt(self, inherited_texts):
        system = (
            "You generate fraudulent news (fake news) in Portuguese, "
            "in the same style as the examples provided."
        )
        if inherited_texts:
            examples = "\n\n".join(
                f"Example {i + 1}:\n{t[:1200]}"
                for i, t in enumerate(inherited_texts[:5])
            )
            user = f"{examples}\n\nGenerate a new fraudulent news article in Portuguese:"
        else:
            user = "Generate a plausible fraudulent news article in Portuguese:"
        return system, user

    def generate_batch(self, inherited_texts, n):
        system, user = self._prompt(inherited_texts)
        batch = []
        for _ in range(n):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=256,
                temperature=0.9,
            )
            batch.append(resp.choices[0].message.content or "")
            self.tokens_used += resp.usage.total_tokens
        return batch
