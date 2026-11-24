"""Paper Fig. 1 (A10) — F1 e PR-AUC média±IC por braço, n=10, 11 cenários.

Lê os grupos revision-controlled-* e o braço Real (real-baseline-*) e gera
figures/revision/fig_controlled_arms.png (painel único, rótulos limpos).
Substitui a figura antiga de Pareto.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analyze_normalized import SCENARIOS, ARMS, load_group

OUT = Path("figures/revision")
OUT.mkdir(parents=True, exist_ok=True)


def ms(xs):
    m = statistics.mean(xs)
    # 95% CI via normal approx (n=10); honest and cheap
    se = statistics.pstdev(xs) / np.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return m, 1.96 * se


def real_values():
    d = {}
    for p in (Path("outputs")).glob("real-baseline-*/real-s*/results.json"):
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if "f1" in r.get("test", {}):
            d.setdefault(r["scenario"], []).append(r["test"])
    return d


def main():
    labels = []
    for scenario, (prompt, ds, cen) in SCENARIOS.items():
        labels.append((scenario, f"{ds}\n{cen}"))
    scenarios = [s for s, _ in labels]
    arms = list(ARMS) + ["real"]
    real = real_values()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, ylab in [(axes[0], "f1", "F1 (teste)"), (axes[1], "pr_auc", "PR-AUC (teste)")]:
        x = np.arange(len(scenarios))
        w = 0.17
        colors = {"all": "#d62728", "random-k": "#1f77b4", "score-k": "#2ca02c",
                  "easy-k": "#ff7f0e", "real": "#7f7f7f"}
        for j, arm in enumerate(arms):
            vals = []
            for scenario in scenarios:
                if arm == "real":
                    ts = real.get(scenario, [])
                    xs = [t[metric] for t in ts if metric in t]
                else:
                    group = f"revision-controlled-{scenario}-{SCENARIOS[scenario][0]}"
                    ts = [r["test"] for r in load_group(group) if r.get("arm") == arm and metric in r.get("test", {})]
                    xs = [t[metric] for t in ts]
                m, ci = ms(xs) if xs else (float("nan"), 0.0)
                vals.append((m, ci))
            ax.bar(x + (j - 2) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals],
                   capsize=2, label=arm, color=colors[arm], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([l[1] for l in labels], fontsize=6.5, rotation=0)
        ax.set_ylabel(ylab)
        ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.suptitle("Estudo controlado (n=10, 11 cenários) — F1 e PR-AUC por regra", fontsize=10)
    fig.tight_layout()
    dest = OUT / "fig_controlled_arms.png"
    fig.savefig(dest, dpi=150)
    print("saved", dest)


if __name__ == "__main__":
    main()
