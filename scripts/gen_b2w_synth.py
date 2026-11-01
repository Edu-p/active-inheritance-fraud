"""Generate the free synthetic NEGATIVE-review pool for the B2W decisive task (API).

Fictional negative product reviews in PT-BR, for sentiment-analysis research.
Output: outputs/decisive/b2w_synth.jsonl

Usage:
  uv run python scripts/gen_b2w_synth.py --n 640 --model openai/gpt-4.1-mini
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

SYSTEM = (
    "Gere uma avaliação NEGATIVA FICTÍCIA de produto em português brasileiro, para "
    "pesquisa de análise de sentimento. Retorne APENAS a avaliação, sem aspas, sem "
    "comentários, em linguagem natural de e-commerce (reclamação, frustração, defeito)."
)
PRODUCTS = ("celular", "notebook", "geladeira", "fone de ouvido", "smart TV", "aspirador", "panela", "tênis")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=640)
    ap.add_argument("--model", default="openai/gpt-4.1-mini")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    load_dotenv()
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPEN_ROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "b2w-synth"},
    )

    dest = Path("outputs/decisive/b2w_synth.jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in dest.read_text().splitlines()) if dest.exists() else 0

    def one(i):
        r = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"Produto: {PRODUCTS[i % len(PRODUCTS)]}. Gere a avaliação negativa fictícia:"}],
            temperature=0.9, max_tokens=120,
        )
        return (r.choices[0].message.content or "").strip()

    todo = list(range(done, args.n))
    print(f"todo {len(todo)}", flush=True)
    with open(dest, "a") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, t in zip(todo, pool.map(one, todo)):
                if t:
                    f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                    f.flush()
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{args.n}", flush=True)
    print("->", dest)


if __name__ == "__main__":
    main()
