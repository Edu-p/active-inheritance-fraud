"""A1 audit — LLM-as-judge of the INHERITED synthetic sets per rule.

Reads the controlled study's stored `synthetic` (inherited texts) per arm and asks
a strong LLM whether each is actually fabricated misinformation ("falsa") or a
plausible true fact ("verdadeira"). Since every synthetic candidate is presumed
fake, the wrong-label fraction per arm = share judged "verdadeira".

Prediction (advisor A1): the score inherits ~2x more wrong-label candidates than
the base rate; Easy-k inherits fewer.

API-GATED (OpenRouter). Resumable (skips existing output file).

Usage:
  uv run python scripts/label_audit.py --scenarios fakebr-10pct,factckbr-10pct,covid19br-10pct,fakewhatsapp-10pct --n 100 --judge-model openai/gpt-4.1-mini
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from scripts.analyze_normalized import SCENARIOS as DISPLAY

ARMS = ("all", "random-k", "score-k", "easy-k")
JUDGE_SYSTEM = (
    "Você é um verificador de fatos. Classifique a mensagem abaixo como "
    "'falsa' (desinformação/boato inventado) ou 'verdadeira' (afirmação factual "
    "plausível). Responda APENAS com uma palavra: 'falsa' ou 'verdadeira'."
)


def load_inherited(scenario):
    prompt = DISPLAY[scenario][0]
    group = Path("outputs") / f"revision-controlled-{scenario}-{prompt}"
    by_arm = {a: [] for a in ARMS}
    for p in group.glob("*/results.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        arm = d.get("arm")
        if arm in by_arm:
            by_arm[arm].extend(d.get("synthetic", []))
    # dedup, keep order
    return {a: list(dict.fromkeys(t for t in texts if t and t.strip())) for a, texts in by_arm.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="fakebr-10pct,factckbr-10pct,covid19br-10pct,fakewhatsapp-10pct")
    ap.add_argument("--n", type=int, default=100, help="samples judged per arm per scenario")
    ap.add_argument("--judge-model", default="openai/gpt-4.1-mini")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
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
        raise ValueError("Set OPEN_ROUTER_API_KEY / GPT_API_KEY / OPENAI_API_KEY")
    kwargs = {"api_key": api_key, "base_url": base_url}
    if base_url and "openrouter.ai" in base_url:
        kwargs["default_headers"] = {"HTTP-Referer": "http://localhost", "X-Title": "label-audit"}
    client = OpenAI(**kwargs)

    outdir = Path("outputs/label-audit")
    outdir.mkdir(parents=True, exist_ok=True)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    for scenario in scenarios:
        dest = outdir / f"{scenario}.json"
        if dest.exists():
            print(f"SKIP {scenario} (already audited)", flush=True)
            continue
        by_arm = load_inherited(scenario)
        verdicts = {a: [] for a in ARMS}
        samples = {}
        for a in ARMS:
            pool = by_arm[a]
            samples[a] = pool[: min(args.n, len(pool))]
        total = sum(len(v) for v in samples.values())
        done = 0
        for a in ARMS:
            for text in samples[a]:
                resp = client.chat.completions.create(
                    model=args.judge_model,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": text[:2000]}],
                    temperature=0.0, max_tokens=8,
                )
                v = (resp.choices[0].message.content or "").strip().lower()
                verdicts[a].append(v)
                done += 1
                if done % 50 == 0:
                    print(f"  {scenario}: {done}/{total}", flush=True)
        result = {
            "scenario": scenario, "judge_model": args.judge_model, "n_per_arm": args.n,
            "wrong_frac": {a: (sum(1 for v in vs if v.startswith("verdadeira")) / len(vs)) if vs else None
                           for a, vs in verdicts.items()},
            "n_judged": {a: len(vs) for a, vs in verdicts.items()},
            "verdicts": verdicts,
        }
        dest.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"DONE {scenario}: wrong_frac={result['wrong_frac']}", flush=True)


if __name__ == "__main__":
    main()
