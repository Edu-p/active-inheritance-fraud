"""Writes a timestamped markdown analysis of the 2 new datasets (COVID19.BR, FACTCK.BR).

Read-only: reads the revision-controlled groups for the new scenarios and emits
docs/analise_novos_datasets_<UTCtimestamp>.md (never overwrites).
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ARMS = ("all", "random-k", "score-k", "easy-k")
SCENARIOS = ("covid19br-full", "covid19br-10pct", "covid19br-5pct",
             "factckbr-full", "factckbr-10pct", "factckbr-5pct")


def runs(group):
    base = Path("outputs") / group
    out = []
    for p in base.glob("*/results.json"):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return out


def mean_sd(xs):
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, s


def fmt(m, s):
    return f"{m:.3f}\\pm{s:.3f}" if s else f"{m:.3f}"


def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(f"docs/analise_novos_datasets_{ts}.md")
    lines = []
    add = lines.append

    add("# Análise — dois novos datasets (COVID19.BR e FACTCK.BR)")
    add("")
    add(f"Gerado em `{ts}` (UTC). Leitura apenas; nenhum resultado sobrescrito. "
        "n=3 seeds (42–44) → exploratório; Wilcoxon bilateral não atinge p<0,05.")
    add("")
    add("Metodologia idêntica à revisão 21/out v2 (`scripts/controlled_curatorship.py`): "
        "uma geração zero-shot por cenário/seed (`candidates.json`) e 4 braços offline "
        "(`all`, `random-k`, `score-k`, `easy-k`) sobre os **mesmos** batches. Treino final "
        "do zero em D0∪H, class weighting e limiar ajustado na validação em todos os braços.")
    add("")
    add("## Datasets")
    add("")
    add("- **COVID19.BR** — mensagens de WhatsApp sobre COVID-19 (PT-BR). Split temporal: "
        "treino=2020-03/04, validação=2020-06-08..09, teste=2020-06-10..12. `fake` é a classe "
        "minoritária (≈1:2.5). Fonte: Zenodo 5193932.")
    add("- **FACTCK.BR** — claims fact-checked (PT-BR). Split **aleatório estratificado** "
        "70/15/15 (seed 42); split temporal evitado porque os rótulos `verdadeiro` se concentram "
        "em 2016–2018 (2019 ficaria sem classe negativa). `fake` é majoritária (corpus de "
        "checagem, que checa majoritariamente claims falsas). Texto = `claimReviewed`; "
        "`reviewBody` excluído (vazaria o veredito). Binarização estrita: falso/falsa→fake, "
        "verdadeiro/verdadeira→true, demais rótulos descartados.")
    add("")
    add("| Dataset | Cenário | Braço | F1 (média±dp) | Recall | PR-AUC | Limiar | n_synth |")
    add("|---|---|---|---|---|---|---|---|")
    for s in SCENARIOS:
        prompt = "v4" if s.startswith("covid19br") else "v2"
        group = f"revision-controlled-{s}-{prompt}"
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
            add(f"| {s.split('-')[0]} | {s.split('-')[1]} | {arm} | {fmt(mf, sf)} | "
                f"{mean_sd(rec)[0]:.3f} | {mean_sd(pr)[0]:.3f} | {mean_sd(th)[0]:.2f} | "
                f"{int(statistics.mean(ns))} |")
    add("")
    add("## Figuras (timestamp, em `figures/revision/`)")
    add("")
    figs = sorted(Path("figures/revision").glob("*new*"))
    if not figs:
        figs = sorted(Path("figures/revision").glob("*.png"))
    for f in figs:
        add(f"- `{f}`")
    if not figs:
        add("- (nenhuma figura ainda; rodar `scripts/diagnose_and_plot.py`)")
    add("")
    add("## Limitações desta extensão")
    add("")
    add("- FACTCK.BR usa split aleatório (não temporal) e tem prior invertido (fake majoritária); "
        "não comparar diretamente com FakeWhatsApp.Br/COVID19.BR sem ressalvar.")
    add("- FACTCK.BR é pequeno (≈1.0k claims após binarização); a classe `true` no teste é "
        "escassa (≈18 exemplos).")
    add("- COVID19.BR não tem dados em maio/2020; o teste é um futuro curto (junho).")
    add("- n=3 seeds; não declarar significância estatística.")
    add("")
    out.write_text("\n".join(lines))
    print(f"Escrito: {out}")


if __name__ == "__main__":
    main()
