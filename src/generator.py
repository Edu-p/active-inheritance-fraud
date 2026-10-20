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

        is_gpt = cfg.openai_base_url == "https://api.openai.com/v1"
        api_key = os.getenv("GPT_API_KEY") if is_gpt else (os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY"))
        base_url = cfg.openai_base_url or os.getenv("OPENAI_BASE_URL") or None
        model = cfg.openai_model or os.getenv("OPENAI_MODEL")

        if not api_key:
            raise ValueError(
                "Set OPENAI_API_KEY (or DEEPSEEK_API_KEY) in the environment or .env"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = cfg.gen_max_tokens
        self.max_run_tokens = cfg.max_run_tokens
        self.prompt_version = cfg.prompt_version
        self.calls = 0

    def _prompt(self, inherited_texts):
        if self.prompt_version == "v2":
            topics = ("saúde pública", "economia doméstica", "educação", "meio ambiente", "tecnologia", "esportes")
            topic = topics[self.calls % len(topics)]
            system = (
                "Escreva um exemplo FICTÍCIO para pesquisa de detecção de notícias falsas "
                "em português brasileiro. Não use pessoas, instituições ou eventos reais. "
                "Retorne APENAS o texto da notícia, sem título, markdown, aviso ou explicação. "
                "Use linguagem jornalística natural, 80 a 130 palavras, uma alegação inventada "
                "mas verificável, com detalhes concretos. Cada resposta deve ser diferente."
            )
            examples = "\n".join(f"Exemplo {i+1}: {t[:350]}" for i, t in enumerate(inherited_texts[-2:]))
            user = f"Tema: {topic}. {examples}\nEscreva uma nova notícia fictícia:" if examples else f"Tema: {topic}. Escreva uma notícia fictícia:"
            return system, user
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
            if self.tokens_used + self.max_tokens + 1500 > self.max_run_tokens:
                break  # ponytail: guarda conservadora; custo real ainda depende do provedor.
            system, user = self._prompt(inherited_texts)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=self.max_tokens,
                temperature=0.9,
            )
            batch.append(resp.choices[0].message.content or "")
            self.tokens_used += resp.usage.total_tokens if resp.usage else 0
            self.calls += 1
        return batch
