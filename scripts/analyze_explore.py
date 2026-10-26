"""Agrega os resultados da exploração (read-only) em tabelas.

Usage: uv run python scripts/analyze_explore.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path("outputs")


def load(group):
    rows = []
    for p in (OUT / group).glob("*/results.json"):
        try:
            rows.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return rows


def mean_sd(xs):
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, s


def main():
    comp = load("explore-competence")
    if comp:
        print("=== competence sweep: ganho score-k sobre random-k (F1) ===")
        print(f"{'scenario':<20} {'frac':>6} {'random':>8} {'score':>8} {'delta':>8}")
        by = {}
        for r in comp:
            by.setdefault((r["scenario"], r["fraction"], r["arm"]), []).append(r["test"]["f1"])
        scen = sorted({k[0] for k in by})
        fracs = sorted({k[1] for k in by})
        for s in scen:
            for f in fracs:
                rnd = by.get((s, f, "random-k"), [])
                scr = by.get((s, f, "score-k"), [])
                if not rnd or not scr:
                    continue
                rm = mean_sd(rnd)[0]
                sm = mean_sd(scr)[0]
                print(f"{s:<20} {f:>6.2f} {rm:>8.3f} {sm:>8.3f} {sm-rm:>+8.3f}")

    band = load("explore-band")
    if band:
        print("\n=== band-k vs score-k/random-k (fração 1.00) ===")
        print(f"{'scenario':<20} {'arm':<10} {'F1 mean':>8} {'F1 sd':>8} {'n':>3}")
        by = {}
        for r in band:
            by.setdefault((r["scenario"], r["arm"]), []).append(r["test"]["f1"])
        for (s, a), xs in sorted(by.items()):
            m, sd = mean_sd(xs)
            print(f"{s:<20} {a:<10} {m:>8.3f} {sd:>8.3f} {len(xs):>3}")

    if not comp and not band:
        print("Sem resultados ainda em explore-competence / explore-band.")


if __name__ == "__main__":
    main()
