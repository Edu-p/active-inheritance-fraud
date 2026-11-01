"""T2.3: LLM-as-judge relabel of the assumed-fake synthetic candidates.

Samples synthetic texts from the frozen candidates.json and asks a strong LLM to
judge fake-vs-plausible. Reports the agreement with the assumed label (all fake).
A low agreement = the generator hallucinates plausible-true facts; a high
agreement = the label assumption holds. Cheaper than human validation.

API-GATED: needs a judge key/model (OpenRouter by default). Do NOT run without confirmation.

Usage:
  uv run python scripts/relabel_judge.py \
    --scenario covid19br-full --seed 42 --n 100 --judge-model google/gemini-3.8-flash
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from scripts.controlled_curatorship import SCENARIOS

CAND_DIR = Path("outputs/revision-candidates")
KIND_PROMPT = {"whatsapp": "v4", "news": "v2", "fakebr": "v2"}

JUDGE_SYSTEM = (
    "Você é um verificador de fatos. Classifique a mensagem abaixo como "
    "'falsa' (desinformação/boato inventado) ou 'verdadeira' (afirmação factual "
    "plausível). Responda APENAS com uma palavra: 'falsa' ou 'verdadeira'."
)


def find_candidates(scenario, seed):
    prompt = KIND_PROMPT[SCENARIOS[scenario][3]]
    for p in sorted(CAND_DIR.glob(f"{scenario}-s{seed}-*/candidates.json")):
        try:
            meta = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if meta.get("complete") and meta.get("zero_shot") and meta.get("prompt") == prompt:
            return p
    raise FileNotFoundError(f"No canonical candidates for {scenario} s{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=100, help="candidates to judge")
    ap.add_argument("--judge-model", default="google/gemini-3.8-flash")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="else env OPENAI_BASE_URL")
    args = ap.parse_args()

    load_dotenv()
    from openai import OpenAI

    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    if base_url and "openrouter.ai" in base_url:
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
    elif base_url and "openai.com" in base_url:
        api_key = os.getenv("GPT_API_KEY")
    else:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Set a judge API key (OPEN_ROUTER_API_KEY / GPT_API_KEY / OPENAI_API_KEY)")
    kwargs = {"api_key": api_key, "base_url": base_url}
    if base_url and "openrouter.ai" in base_url:
        kwargs["default_headers"] = {"HTTP-Referer": "http://localhost", "X-Title": "active-inheritance-fraud"}
    client = OpenAI(**kwargs)

    cand = find_candidates(args.scenario, args.seed)
    meta = json.loads(cand.read_text())
    texts = [t for b in meta["batches"] for t in b if t.strip()]
    texts = texts[: args.n]

    verdicts = []
    for i, text in enumerate(texts):
        resp = client.chat.completions.create(
            model=args.judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": text[:2000]},
            ],
            temperature=0.0,
            max_tokens=8,
        )
        v = (resp.choices[0].message.content or "").strip().lower()
        verdicts.append(v)
        print(f"{i + 1}/{len(texts)} -> {v!r}", flush=True)

    n_fake = sum(1 for v in verdicts if v.startswith("falsa"))
    out = {
        "scenario": args.scenario, "seed": args.seed, "judge_model": args.judge_model,
        "n": len(texts), "n_judged_fake": n_fake,
        "agreement_with_assumed_fake": n_fake / len(texts) if texts else 0.0,
        "verdicts": verdicts,
    }
    outdir = Path("outputs/relabel-judge")
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"{args.scenario}-s{args.seed}.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"agreement={out['agreement_with_assumed_fake']:.2f} -> {dest}")


if __name__ == "__main__":
    main()
