"""Synthetic text generators.

- MockGenerator: perturbs real fake news, so the pipeline runs offline.
- OpenAIGenerator: uses an LLM (OpenAI-compatible API).
"""
from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv


class BaseGenerator:
    def __init__(self):
        self.tokens_used = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0

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
        self.workers = cfg.generator_workers
        self.zero_shot = getattr(cfg, "zero_shot", False)
        self.calls = 0
        # ponytail: preços aproximados USD/1M tokens (in, out); recalibrar se o provedor mudar.
        self._price_in = {"deepseek-chat": 0.27, "deepseek-flash": 0.07, "gpt-4o-mini": 0.15, "gpt-4o": 5.0}
        self._price_out = {"deepseek-chat": 1.10, "deepseek-flash": 0.27, "gpt-4o-mini": 0.60, "gpt-4o": 15.0}

    def _add_cost(self, prompt_tokens, completion_tokens):
        p_in = self._price_in.get(self.model, 0.0)
        p_out = self._price_out.get(self.model, 0.0)
        self.cost_usd += prompt_tokens / 1e6 * p_in + completion_tokens / 1e6 * p_out

    def _prompt(self, inherited_texts, call_index=None):
        if getattr(self, "zero_shot", False):
            inherited_texts = []
        call_index = self.calls if call_index is None else call_index
        if self.prompt_version == "v4":
            topics = ("saúde", "eleições", "segurança", "economia", "educação", "serviços públicos")
            styles = ("dúvida informal", "relato pessoal", "conversa curta", "aviso comunitário", "pedido de checagem", "mensagem de grupo")
            lengths = ((20, 60), (61, 120), (121, 200), (201, 350))
            lo, hi = lengths[call_index % len(lengths)]
            system = (
                "Gere uma mensagem FICTÍCIA de desinformação para pesquisa de detecção em WhatsApp "
                "brasileiro. Não use pessoas, instituições, links ou eventos reais. Retorne APENAS "
                "a mensagem, sem título, Markdown, aviso ou explicação. Evite aberturas repetitivas, "
                "especialmente 'URGENTE' e 'acabei de receber'. Não copie outros textos."
            )
            user = f"Tema: {topics[call_index % len(topics)]}. Estilo: {styles[call_index % len(styles)]}. Comprimento: {lo} a {hi} palavras. Escreva uma mensagem inventada natural."
            return system, user
        if self.prompt_version == "v3":
            topics = ("saúde", "eleições", "segurança", "economia", "serviços públicos", "educação")
            topic = topics[call_index % len(topics)]
            system = (
                "Gere uma mensagem FICTÍCIA de desinformação para pesquisa acadêmica de detecção "
                "em WhatsApp brasileiro. Não use pessoas, instituições, links ou eventos reais. "
                "Retorne APENAS a mensagem, sem título, Markdown, aviso ou explicação. Escreva "
                "40 a 120 palavras, em estilo natural de encaminhamento, com uma alegação inventada "
                "e detalhes concretos. Varie tom, abertura e estrutura; não copie os exemplos."
            )
            examples = "\n".join(f"Exemplo de estilo {i+1}: {t[:350]}" for i, t in enumerate(inherited_texts[-2:]))
            user = f"Tema: {topic}.\n{examples}\nNova mensagem fictícia:" if examples else f"Tema: {topic}. Nova mensagem fictícia:"
            return system, user
        if self.prompt_version == "v2":
            topics = ("saúde pública", "economia doméstica", "educação", "meio ambiente", "tecnologia", "esportes")
            topic = topics[call_index % len(topics)]
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

    def _complete(self, inherited_texts, call_index):
        system, user = self._prompt(inherited_texts, call_index)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=self.max_tokens,
            temperature=0.9,
        )
        usage = resp.usage
        return (
            resp.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    def generate_batch(self, inherited_texts, n):
        batch = []
        while len(batch) < n:
            system, user = self._prompt(inherited_texts, self.calls)
            # ponytail: conservative per-call bound; process chunks so the run stays near its cap.
            estimated_max = self.max_tokens + max(128, (len(system) + len(user)) // 2)
            slots = (self.max_run_tokens - self.tokens_used) // estimated_max
            count = min(self.workers, n - len(batch), max(0, slots))
            if count == 0:
                break
            indices = range(self.calls, self.calls + count)
            if count == 1:
                results = [self._complete(inherited_texts, self.calls)]
            else:
                with ThreadPoolExecutor(max_workers=count) as pool:
                    results = list(pool.map(lambda i: self._complete(inherited_texts, i), indices))
            batch.extend(text for text, _, _ in results)
            prompt = sum(pt for _, pt, _ in results)
            completion = sum(ct for _, _, ct in results)
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.tokens_used += prompt + completion
            self._add_cost(prompt, completion)
            self.calls += count
        return batch
