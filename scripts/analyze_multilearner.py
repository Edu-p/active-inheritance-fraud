"""Aggregate the multi-learner replay (T2.2) into a table.

Reads outputs/multilearner-* (warm-start replay from controlled_curatorship).
Groups by (scenario, learner_model) and reports random-k vs score-k, delta,
and a paired Wilcoxon p across seeds.

Usage:
  uv run python scripts/analyze_multilearner.py
  uv run python scripts/analyze_multilearner.py --latex
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from scipy.stats import wilcoxon

OUT = Path("outputs")


def load(group):
    rows = []
    for p in (OUT / group).glob("*/results.json"):
        try:
            rows.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return rows


def ms(xs):
    return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    data = {}  # (scenario, learner) -> {arm: {seed: f1}}
    for group in sorted(p.name for p in OUT.glob("multilearner-*")):
        scenario = group[len("multilearner-"):].rsplit("-", 1)[0]  # strip model suffix
        for r in load(group):
            model = r.get("config", {}).get("learner_model", "?")
            arm = r.get("arm")
            seed = int(r.get("config", {}).get("random_state", -1))
            if arm not in ("score-k", "random-k") or "f1" not in r.get("test", {}):
                continue
            data.setdefault((scenario, model), {}).setdefault(arm, {})[seed] = r["test"]["f1"]

    if not data:
        print("Nenhum resultado multilearner ainda.")
        return

    print(f"{'scenario':<18} {'learner':<38} {'random':>14} {'score':>14} {'Δ':>8} {'p':>7} {'n':>3}")
    for (scenario, model), arms in sorted(data.items()):
        rnd = arms.get("random-k", {})
        scr = arms.get("score-k", {})
        seeds = sorted(set(rnd) & set(scr))
        if len(seeds) < 1:
            continue
        mr, sr = ms([rnd[s] for s in seeds])
        msc, ssc = ms([scr[s] for s in seeds])
        delta = statistics.mean([scr[s] - rnd[s] for s in seeds])
        p = None
        if len(seeds) >= 2:
            _, p = wilcoxon([scr[s] for s in seeds], [rnd[s] for s in seeds], zero_method="wilcox")
        print(f"{scenario:<18} {model:<38} {mr:>7.3f} ± {sr:<5.3f} {msc:>7.3f} ± {ssc:<5.3f} "
              f"{delta:>+8.3f} {p if p is not None else float('nan'):>7.4f} {len(seeds):>3}")
        if args.latex:
            print(f"LATEX {scenario} & {model} & {mr:.3f} $\\pm$ {sr:.3f} & "
                  f"{msc:.3f} $\\pm$ {ssc:.3f} & {delta:+.3f} & "
                  f"{p if p is not None else float('nan'):.3f} \\\\")


if __name__ == "__main__":
    main()
