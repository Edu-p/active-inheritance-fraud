"""Generate the free synthetic TOXIC pool for the ToLD-Br decisive task (API).

Fictional hate/toxic messages in PT-BR internet language, for toxicity-detection
research. No real people/groups targeted. Output: outputs/decisive/toxic_synth.jsonl

Usage:
  uv run python scripts/gen_toxic_synth.py --n 640 --model openai/gpt-4.1-mini
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
    "Gere uma mensagem TÓXICA/ODIOSA FICTÍCIA para pesquisa de detecção de "
    "toxicidade em português brasileiro. Não direcione a pessoas ou grupos reais; "
    "use alvos genéricos. Retorne APENAS a mensagem, sem aspas, sem comentários, "
    "em linguagem de internet (xingamentos, ofensas, ódio)."
)
TOPICS = ("futebol", "política", "aparência", "gênero", "religião", "classe social")


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
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "toxic-synth"},
    )

    dest = Path("outputs/decisive/toxic_synth.jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    if dest.exists():
        done = sum(1 for _ in dest.read_text().splitlines())

    def one(i):
        r = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"Tema: {TOPICS[i % len(TOPICS)]}. Gere a mensagem tóxica fictícia:"}],
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
