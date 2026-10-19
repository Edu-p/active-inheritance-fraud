"""Testes estatísticos entre braços (Kruskal-Wallis + Wilcoxon) sobre F1.

Comparação pareada por seed. Experimento vem do nome do subdiretório (e1..e5).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, wilcoxon

OUT = Path("outputs")


def load():
    runs = []
    for f in OUT.rglob("results.json"):
        d = json.loads(f.read_text())
        exp = f.parent.parent.name  # e1..e5
        seed = int(d["config"].get("random_state", -1))
        n = d["config"].get("n_samples")
        runs.append({
            "exp": exp,
            "seed": seed,
            "n": int(n) if n not in (None, "None") else None,
            "f1": d["test"]["f1"],
            "tokens": int(d.get("tokens_used") or 0),
        })
    return runs


def mean_std(xs):
    return float(np.mean(xs)), float(np.std(xs))


def by_seed(runs, exp, n=None):
    d = {}
    for r in runs:
        if r["exp"] == exp and (n is None or r["n"] == n):
            d[r["seed"]] = r["f1"]
    return d


def main():
    runs = load()

    print("== resumo por experimento ==")
    for exp in ["e1", "e2", "e3", "e4", "e5"]:
        rs = [r for r in runs if r["exp"] == exp]
        if not rs:
            continue
        m, s = mean_std([r["f1"] for r in rs])
        tok = sum(r["tokens"] for r in rs)
        print(f"  {exp}: n={len(rs)}  F1={m:.4f}±{s:.4f}  tokens_total={tok}")

    # por seed para o e2
    for n in (32, 79, 158):
        rs = [r for r in runs if r["exp"] == "e2" and r["n"] == n]
        if rs:
            m, s = mean_std([r["f1"] for r in rs])
            print(f"    e2 static n={n}: F1={m:.4f}±{s:.4f}")

    real = by_seed(runs, "e1")
    ai = by_seed(runs, "e3")

    print("\n== Wilcoxon (pareado por seed) ==")
    common = sorted(set(real) & set(ai))
    if len(common) >= 2:
        w, p = wilcoxon([real[s] for s in common], [ai[s] for s in common])
        print(f"  e1(real) vs e3(ai): n={len(common)}  W={w:.4f}  p={p:.4f}")

    for n in (32, 79, 158):
        st = by_seed(runs, "e2", n)
        common = sorted(set(st) & set(ai))
        if len(common) >= 2:
            w, p = wilcoxon([st[s] for s in common], [ai[s] for s in common])
            print(f"  e2(static n={n}) vs e3(ai): n={len(common)}  W={w:.4f}  p={p:.4f}")

    # Kruskal-Wallis: real vs melhor-static-por-seed vs ai
    best_static = {}
    for r in runs:
        if r["exp"] == "e2":
            best_static[r["seed"]] = max(best_static.get(r["seed"], -1), r["f1"])
    groups = {
        "real": [real[s] for s in sorted(real)],
        "static_best": [best_static[s] for s in sorted(best_static)],
        "ai": [ai[s] for s in sorted(ai)],
    }
    if all(len(g) >= 2 for g in groups.values()):
        h, p = kruskal(*groups.values())
        print(f"\nKruskal-Wallis (real vs static-melhor vs ai): H={h:.4f}  p={p:.4f}")

    print("\nNota: n=3 seeds → baixo poder estatístico. Resultados são indicativos.")


if __name__ == "__main__":
    main()
