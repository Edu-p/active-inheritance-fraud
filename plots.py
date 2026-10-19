"""Gera as figuras da entrega 2 a partir de outputs/*/results.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs")
FIG = Path("figures")


def load_runs():
    runs = []
    for f in sorted(OUT.rglob("results.json")):
        runs.append(json.loads(f.read_text()))
    return runs


def _arm(r):
    return r.get("arm", r.get("config", {}).get("arm", "?"))


def _seed(r):
    return r.get("config", {}).get("random_state", "?")


def _tokens(r):
    if r.get("tokens_used"):
        return r["tokens_used"]
    h = r.get("history") or [{}]
    return h[-1].get("tokens_used", 0)


def _f1(r):
    return r.get("test", {}).get("f1")


def fig_f1_vs_tokens(runs):
    plt.figure()
    for arm, color in [("real", "k"), ("static", "C1"), ("ai", "C0")]:
        xs = [_tokens(r) for r in runs if _arm(r) == arm]
        ys = [_f1(r) for r in runs if _arm(r) == arm]
        if xs:
            plt.scatter(xs, ys, label=arm, color=color)
    plt.xscale("log")
    plt.xlabel("tokens acumulados (log)")
    plt.ylabel("F1 (teste)")
    plt.title("F1 x tokens (Static vs ActiveInheritance)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig_f1_vs_tokens.png", dpi=300)
    plt.close()


def fig_f1_per_cycle(runs):
    plt.figure()
    for r in runs:
        if _arm(r) != "ai":
            continue
        h = r.get("history", [])
        plt.plot([e["epoch"] for e in h], [e["val"]["f1"] for e in h],
                 marker="o", label=f"seed {_seed(r)}")
    plt.xlabel("ciclo t")
    plt.ylabel("F1 (validacao)")
    plt.title("ActiveInheritance: F1 por ciclo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig_f1_per_cycle.png", dpi=300)
    plt.close()


def fig_wasserstein(runs):
    vals, labels = [], []
    for arm, color in [("static", "C1"), ("ai", "C0")]:
        ws = [r.get("quality", {}).get("fidelity", {}).get("wasserstein")
              for r in runs if _arm(r) == arm]
        ws = [w for w in ws if w is not None]
        if ws:
            vals.append(sum(ws) / len(ws))
            labels.append(arm)
    if not vals:
        return
    plt.figure()
    plt.bar(labels, vals, color=["C1", "C0"][: len(vals)])
    plt.ylabel("Wasserstein medio (PCA)")
    plt.title("Fidelidade real x sintetico")
    plt.tight_layout()
    plt.savefig(FIG / "fig_wasserstein.png", dpi=300)
    plt.close()


def fig_diversity_cycle(runs):
    plt.figure()
    for r in runs:
        if _arm(r) != "ai":
            continue
        h = r.get("history", [])
        plt.plot([e["epoch"] for e in h],
                 [e["pool_diversity"]["mean_pairwise_sim"] for e in h],
                 marker="o", label=f"seed {_seed(r)}")
    plt.xlabel("ciclo t")
    plt.ylabel("sim media do pool")
    plt.title("Diversidade do pool por ciclo (colapso modal)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig_diversity_cycle.png", dpi=300)
    plt.close()


def fig_f1_boxplot(runs):
    data = {}
    for arm in ("real", "static", "ai"):
        fs = [_f1(r) for r in runs if _arm(r) == arm]
        if fs:
            data[arm] = fs
    if not data:
        return
    plt.figure()
    plt.boxplot(list(data.values()))
    plt.xticks(range(1, len(data) + 1), list(data.keys()))
    plt.ylabel("F1 (teste)")
    plt.title("Consistencia entre seeds")
    plt.tight_layout()
    plt.savefig(FIG / "fig_f1_boxplot.png", dpi=300)
    plt.close()


def fig_score_decomp(runs):
    for r in runs:
        if _arm(r) != "ai":
            continue
        h = r.get("history", [])
        if not h or "score_parts" not in h[0]:
            continue
        plt.figure()
        epochs = [e["epoch"] for e in h]
        for key, label in [("error", "erro"), ("uncertainty", "incerteza"),
                           ("diversity", "diversidade")]:
            plt.plot(epochs, [e["score_parts"][key] for e in h],
                     marker="o", label=label)
        plt.xlabel("ciclo t")
        plt.ylabel("componente medio")
        plt.title(f"Decomposicao do score (seed {_seed(r)})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / "fig_score_decomp.png", dpi=300)
        plt.close()
        break


def main():
    runs = load_runs()
    if not runs:
        print("Nenhum run em outputs/. Rode os experimentos antes.")
        return
    FIG.mkdir(exist_ok=True)
    fig_f1_vs_tokens(runs)
    fig_f1_per_cycle(runs)
    fig_wasserstein(runs)
    fig_diversity_cycle(runs)
    fig_f1_boxplot(runs)
    fig_score_decomp(runs)
    print(f"Figuras salvas em {FIG.resolve()}")


if __name__ == "__main__":
    main()
