"""Decisive v2 — graduated-detectability candidates via LLM paraphrase (API).

Takes REAL held-out fakes and rewrites them at 3 increasing "synthetic register"
strengths, keeping the claims ~fixed (content stays, only voice changes). This
isolates detectability from the wrong-label confound of the alpha-mixture.

Levels:
  p1 (light)    : same claims, same journalistic style, just reworded
  p2 (medium)   : same claims, names/places generalized ("um político", "uma cidade")
  p3 (strong)   : rewritten as a circulating-rumor alert ("Estão dizendo que...")

Pool P0 = raw held-out fakes (AUC ~0.5); P1..P3 interpolate toward the synthetic
register; free LLM generation sits at the top (AUC ~1.0).

API-GATED (OpenRouter). Resumable. Output: outputs/decisive/paraphrases.jsonl

Usage:
  uv run python scripts/decisive_paraphrase.py --n 320 --model openai/gpt-4.1-mini
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv

PROMPTS = {
    "p1": ("Reescreva o texto abaixo mantendo EXATAMENTE as mesmas alegações e fatos, "
           "mudando apenas a formulação (sinônimos, ordem das frases). Mantenha o estilo "
           "jornalístico. Retorne APENAS o texto reescrito, sem aspas, sem comentários."),
    "p2": ("Reescreva o texto abaixo mantendo as mesmas alegações, mas generalize: "
           "substitua nomes próprios de pessoas e lugares por termos genéricos como "
           "'um político', 'uma cidade', 'uma empresa'. Use um tom de boato genérico. "
           "Retorne APENAS o texto reescrito."),
    "p3": ("Reescreva o texto abaixo como um alerta de desinformação que circula em grupos, "
           "começando com frases como 'Estão dizendo que...' ou 'Um texto que circula "
           "afirma que...'. Sem nomes reais. Mantenha as alegações centrais. "
           "Retorne APENAS o texto reescrito."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--model", default="openai/gpt-4.1-mini")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    load_dotenv()
    from openai import OpenAI

    base_url = args.base_url
    api_key = os.getenv("OPEN_ROUTER_API_KEY") if "openrouter.ai" in base_url else os.getenv("OPENAI_API_KEY")
    kwargs = {"api_key": api_key, "base_url": base_url}
    if "openrouter.ai" in base_url:
        kwargs["default_headers"] = {"HTTP-Referer": "http://localhost", "X-Title": "decisive"}
    client = OpenAI(**kwargs)

    held = pd.read_csv("dataset/fakebr-raw/heldout_fakes.csv")["preprocessed_news"].tolist()
    held = held[: args.n]

    outdir = Path("outputs/decisive")
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "paraphrases.jsonl"
    done = set()
    if dest.exists():
        for line in dest.read_text().splitlines():
            try:
                done.add((json.loads(line)["id"], json.loads(line)["level"]))
            except (ValueError, KeyError):
                pass

    def one(item):
        idx, text, level = item
        if (idx, level) in done:
            return None
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": PROMPTS[level]},
                      {"role": "user", "content": text[:4000]}],
            temperature=0.8, max_tokens=600,
        )
        out = (resp.choices[0].message.content or "").strip()
        return {"id": idx, "level": level, "text": out}

    items = [(i, t, lvl) for i, t in enumerate(held) for lvl in PROMPTS]
    todo = [it for it in items if (it[0], it[2]) not in done]
    print(f"todo {len(todo)} / total {len(items)}", flush=True)
    with open(dest, "a") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, r in enumerate(pool.map(one, todo)):
                if r and r["text"]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)
    print("->", dest)


if __name__ == "__main__":
    main()
