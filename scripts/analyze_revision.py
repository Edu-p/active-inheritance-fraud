"""Writes a timestamped markdown analysis of the Cleber revision runs (read-only).

Usage: uv run python scripts/analyze_revision.py
Output: docs/analise_revisao_cleber_<UTCtimestamp>.md (never overwrites).
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ARMS = ("all", "random-k", "score-k", "easy-k")
SCENARIOS = ("fakewhatsapp-full", "fakewhatsapp-10pct", "fakewhatsapp-5pct")
HIST = {
    "fakewhatsapp-full": {"real": 0.676, "static": 0.715, "ai": 0.708},
    "fakewhatsapp-10pct": {"real": 0.075, "static": 0.229, "ai": 0.179},
    "fakewhatsapp-5pct": {"real": 0.000, "static": 0.099, "ai": 0.055},
}


def runs(group):
    base = Path("outputs") / group
    out = []
    for p in base.glob("*/results.json"):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return out


def mean_sd(xs, pop=True):
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if pop and len(xs) > 1 else 0.0
    return m, s


def fmt(m, s):
    return f"{m:.3f}\\pm{s:.3f}" if s else f"{m:.3f}"


def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(f"docs/analise_revisao_cleber_{ts}.md")
    lines = []
    add = lines.append

    add(f"# Análise da revisão — comentários do Cleber (21/out v2)")
    add("")
    add(f"Gerado em `{ts}` (UTC). Leitura apenas; nenhum resultado sobrescrito. "
        f"n=3 seeds (42–44) → exploratório; Wilcoxon bilateral não atinge p<0,05.")
    add("")
    add("Contexto: `planning/REVISAO_CLEBER_21OCT_V2.md`. Painel: "
        "`uv run python scripts/revision_status.py`.")
    add("")

    add("## Fase B — replay weighted + limiar na validação (zero API)")
    add("")
    add("Treina BERTimbau novo (3 épocas) em treino real + pool sintético **já salvo**, "
        "pondera classes e ajusta limiar na validação. Compara com o F1 histórico (limiar 0,5, sem pesos).")
    add("")
    add("| Cenário | Braço | F1 replay (média±dp) | Recall | PR-AUC | Limiar | F1 histórico |")
    add("|---|---|---|---|---|---|---|")
    for s in SCENARIOS:
        group = f"{s}-weighted-replay-21oct-v2"
        for arm in ("real", "static", "ai"):
            rs = [r for r in runs(group) if r.get("arm") == arm]
            if not rs:
                continue
            f1 = [r["test"]["f1"] for r in rs]
            rec = [r["test"]["recall"] for r in rs]
            pr = [r["test"]["pr_auc"] for r in rs]
            th = [r["test"]["threshold"] for r in rs]
            mf, sf = mean_sd(f1)
            add(f"| {s} | {arm} | {fmt(mf, sf)} | {mean_sd(rec)[0]:.3f} | "
                f"{mean_sd(pr)[0]:.3f} | {mean_sd(th)[0]:.2f} | {HIST[s][arm]:.3f} |")
    add("")
    add("Leitura: class weighting + limiar recupera recall sob escassez (Real 10% 0,075→0,621; "
        "5% 0,000→0,642). Isso corrobora a hipótese do professor de que o colapso era em parte "
        "limiar fixo 0,5 + desbalanceamento, não só gate ruim.")
    add("")

    add("## Fase C — estudo controlado de curadoria (mesmos candidatos)")
    add("")
    add("Uma geração zero-shot (prompt v4) por cenário/seed, salva em `candidates.json`; os 4 braços "
        "reaproveitam os **mesmos batches** e diferem só na regra de herança. `all` herda tudo; "
        "`random-k`/`score-k`/`easy-k` herdam top-16 por ciclo (≈80 textos). Treino final do zero em "
        "D0∪H, class weighting e limiar na validação em todos.")
    add("")
    add("| Cenário | Braço | F1 (média±dp) | Recall | PR-AUC | Limiar | n_synth |")
    add("|---|---|---|---|---|---|---|")
    for s in SCENARIOS:
        group = f"revision-controlled-{s}-v4"
        for arm in ARMS:
            rs = [r for r in runs(group) if r.get("arm") == arm]
            if not rs:
                continue
            f1 = [r["test"]["f1"] for r in rs]
            rec = [r["test"]["recall"] for r in rs]
            pr = [r["test"]["pr_auc"] for r in rs]
            th = [r["test"]["threshold"] for r in rs]
            ns = [r.get("n_synthetic", 0) for r in rs]
            mf, sf = mean_sd(f1)
            add(f"| {s} | {arm} | {fmt(mf, sf)} | {mean_sd(rec)[0]:.3f} | "
                f"{mean_sd(pr)[0]:.3f} | {mean_sd(th)[0]:.2f} | {int(statistics.mean(ns))} |")
    add("")
    add("Leitura: com candidatos iguais, `score-k` **não** domina. `random-k` lidera em 10% e 5%, "
        "`score-k` empata no full. O ganho antes atribuído à Eq. 1 não se sustenta quando o controle "
        "aleatório usa o mesmo gerador/ciclos — o efeito principal é curadoria de volume "
        "(≈80 textos > 277) e rebalanceamento/limiar, não o score em si. `easy-k` não supera "
        "consistentemente; a inversão Profile2Gen não foi confirmada.")
    add("")

    add("## Diagnóstico real-vs-sintético (TF-IDF, split aleatório)")
    add("")
    add("Classificador logístico sobre unigramas+bigramas separa textos reais (fake) de sintéticos "
        "com AUC ≈ 1,0 em todos os cenários/braços. É um diagnóstico de atalho estilístico, **não** "
        "veracidade nem generalização temporal. Atenção: o prompt v4 ainda vaza marcadores "
        "(`fictícia`/`inventada`/`aviso`) e aberturas em caixa alta; isso pode inflar a separabilidade. "
        "Validar com checagem humana antes de afirmar realismo.")
    add("")

    add("## Figuras geradas (timestamp, em `figures/revision/`)")
    add("")
    figs = sorted(Path("figures/revision").glob("*.png"))
    if figs:
        for f in figs:
            add(f"- `{f}`")
    else:
        add("- (nenhuma figura ainda; rodar `scripts/diagnose_and_plot.py`)")
    add("")
    add("## Pendências (não resolvidas por código/GPU)")
    add("")
    add("- Fake.br controlado (exige pipeline de pré-processamento exato ou corpus raw).")
    add("- 10 seeds confirmatórias (gate após piloto; custo autorizado).")
    add("- Curadoria humana de veracidade/qualidade dos sintéticos.")
    add("- Trabalhos relacionados e referências verificadas; novo manuscrito/PDF.")
    add("")

    out.write_text("\n".join(lines))
    print(f"Escrito: {out}")


if __name__ == "__main__":
    main()
