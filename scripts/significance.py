"""Paired significance on the controlled study (read-only).

For each scenario: Wilcoxon signed-rank (paired by seed) of score-k vs random-k,
plus Holm-corrected p-values across scenarios. Also reports All vs Random-k
(the volume-curation effect) the same way.

No training, no API. Feeds the paper tables with p-values instead of bare deltas.

Usage:
  uv run python scripts/significance.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

OUT = Path("outputs")

SCENARIOS = {
    "fakewhatsapp-full": ("v4", "FakeWhatsApp.Br"),
    "fakewhatsapp-10pct": ("v4", "FakeWhatsApp.Br"),
    "fakewhatsapp-5pct": ("v4", "FakeWhatsApp.Br"),
    "covid19br-full": ("v4", "COVID19.BR"),
    "covid19br-10pct": ("v4", "COVID19.BR"),
    "covid19br-5pct": ("v4", "COVID19.BR"),
    "factckbr-full": ("v2", "FACTCK.BR"),
    "factckbr-10pct": ("v2", "FACTCK.BR"),
    "factckbr-5pct": ("v2", "FACTCK.BR"),
    "fakebr-10pct": ("v2", "Fake.br"),
    "fakebr-5pct": ("v2", "Fake.br"),
}


def load_group(group):
    rows = []
    for p in (OUT / group).glob("*/results.json"):
        try:
            rows.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return rows


def by_seed(rows):
    """{seed: f1}"""
    return {int(r["config"]["random_state"]): r["test"]["f1"] for r in rows if "f1" in r.get("test", {})}


def holm(ps):
    m = len(ps)
    order = np.argsort(ps)
    adj = np.ones(m)
    for rank, i in enumerate(order):
        adj[i] = min(1.0, ps[i] * (m - rank))
    # enforce monotonicity
    for rank in range(1, m):
        i_prev, i = order[rank - 1], order[rank]
        adj[i] = max(adj[i], adj[i_prev])
    return adj


def paired_test(a_rows, b_rows):
    a, b = by_seed(a_rows), by_seed(b_rows)
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    da = [a[s] for s in seeds]
    db = [b[s] for s in seeds]
    w, p = wilcoxon(da, db, zero_method="wilcox", alternative="two-sided")
    return {"n": len(seeds), "mean_a": statistics.mean(da), "mean_b": statistics.mean(db),
            "delta": statistics.mean([x - y for x, y in zip(da, db)]), "W": float(w), "p": float(p)}


def main():
    results = []
    for scenario, (prompt, ds) in SCENARIOS.items():
        group = f"revision-controlled-{scenario}-{prompt}"
        rows = load_group(group)
        score = [r for r in rows if r.get("arm") == "score-k"]
        random_ = [r for r in rows if r.get("arm") == "random-k"]
        all_ = [r for r in rows if r.get("arm") == "all"]
        t1 = paired_test(score, random_)
        t2 = paired_test(all_, random_)
        results.append((ds, scenario, t1, t2))

    # Holm across the score-vs-random cells
    ps = [r[2]["p"] for r in results if r[2]]
    adj = holm(ps) if ps else []
    idx = 0
    print(f"{'Domínio':<16} {'Cen.':<8} {'n':>3} {'score':>7} {'random':>7} {'Δ':>8} {'p':>8} {'p_holm':>8}  | {'All p':>8}")
    for ds, scenario, t1, t2 in results:
        cen = scenario.split("-", 1)[1]
        if t1 is None:
            print(f"{ds:<16} {cen:<8}   -- (fewer than 2 paired seeds)")
            continue
        print(f"{ds:<16} {cen:<8} {t1['n']:>3} {t1['mean_a']:>7.3f} {t1['mean_b']:>7.3f} "
              f"{t1['delta']:>+8.3f} {t1['p']:>8.4f} {adj[idx]:>8.4f}  | "
              f"{t2['p'] if t2 else float('nan'):>8.4f}")
        idx += 1

    print("\nInterpretação: Δ = score − random (média pareada por seed). "
          "p_holm corrigido sobre as células score-vs-random.")


if __name__ == "__main__":
    main()
