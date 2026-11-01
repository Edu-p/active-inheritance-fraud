"""Aggregate the ablation groups (weight sweep + style-blind) into tables.

Reads outputs/ablation-weights-* and outputs/ablation-styleblind-*. Paired by
seed (all arms share the same frozen-scorer random-k control). Prints console
tables; --latex emits booktabs bodies.

Usage:
  uv run python scripts/analyze_ablations.py
  uv run python scripts/analyze_ablations.py --latex
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

OUT = Path("outputs")
ARMS = ["random-k", "score-w040_040_020", "score-w100_000_000", "score-w000_100_000",
        "score-w000_000_100", "score-w050_050_000", "score-w033_033_033"]


def load(group):
    rows = []
    for p in (OUT / group).glob("*/results.json"):
        try:
            rows.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return rows


def by_seed(rows, arm):
    return {int(r["seed"]): r["test"]["f1"] for r in rows if r.get("arm") == arm and "f1" in r.get("test", {})}


def ms(xs):
    return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def paired_p(a, b):
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    da = [a[s] for s in seeds]
    db = [b[s] for s in seeds]
    _, p = wilcoxon(da, db, zero_method="wilcox")
    return float(p)


def scenario_groups(mode):
    return sorted(p.name for p in OUT.glob(f"ablation-{mode}-*"))


def weights_table(latex):
    print(f"\n{'='*90}\nWEIGHT ABLATION (frozen-scorer protocol, paired by seed)\n{'='*90}")
    for group in scenario_groups("weights"):
        scenario = group[len("ablation-weights-"):]
        rows = load(group)
        rnd = by_seed(rows, "random-k")
        if not rnd:
            continue
        print(f"\n{scenario}  (n={len(rnd)} seeds)")
        hdr = f"{'arm':<22} {'F1':>14} {'Δ vs random':>12} {'p (paired)':>11}"
        print(hdr)
        for arm in ARMS:
            if arm == "random-k":
                continue
            a = by_seed(rows, arm)
            if not a:
                continue
            m, s = ms(list(a.values()))
            seeds = sorted(set(a) & set(rnd))
            delta = statistics.mean([a[s] - rnd[s] for s in seeds])
            p = paired_p(a, rnd)
            print(f"{arm:<22} {m:.3f} ± {s:.3f} {delta:>+12.3f} {p if p is not None else float('nan'):>11.4f}")
        if latex:
            for arm in ARMS:
                if arm == "random-k":
                    continue
                a = by_seed(rows, arm)
                if not a:
                    continue
                m, s = ms(list(a.values()))
                seeds = sorted(set(a) & set(rnd))
                delta = statistics.mean([a[s] - rnd[s] for s in seeds])
                p = paired_p(a, rnd)
                print(f"LATEX {scenario} & {arm} & {m:.3f} $\\pm$ {s:.3f} & {delta:+.3f} & {p if p is not None else float('nan'):.3f} \\\\")


def styleblind_table(latex):
    print(f"\n{'='*90}\nSTYLE-BLIND SCORE (SemanticProbe scorer, frozen-scorer protocol)\n{'='*90}")
    for group in scenario_groups("styleblind"):
        scenario = group[len("ablation-styleblind-"):]
        rows = load(group)
        rnd = by_seed(rows, "random-k")
        sb = by_seed(rows, "score-styleblind")
        if not rnd or not sb:
            print(f"\n{scenario}: no complete results yet")
            continue
        seeds = sorted(set(rnd) & set(sb))
        delta = statistics.mean([sb[s] - rnd[s] for s in seeds])
        p = paired_p(sb, rnd)
        mr, sr = ms(list(rnd.values()))
        mb, sb_ = ms(list(sb.values()))
        print(f"\n{scenario} (n={len(seeds)}): random={mr:.3f}±{sr:.3f}  "
              f"styleblind={mb:.3f}±{sb_:.3f}  Δ={delta:+.3f}  p={p if p is not None else float('nan'):.4f}")
        if latex:
            print(f"LATEX {scenario} & {mr:.3f} $\\pm$ {sr:.3f} & {mb:.3f} $\\pm$ {sb_:.3f} & {delta:+.3f} & {p if p is not None else float('nan'):.3f} \\\\")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    weights_table(args.latex)
    styleblind_table(args.latex)


if __name__ == "__main__":
    main()
