"""Build the consolidated FakeWhatsApp.Br report from immutable result JSONs."""
from __future__ import annotations

import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import wilcoxon

GROUPS = ["fakewhatsapp-full", "fakewhatsapp-10pct", "fakewhatsapp-5pct"]
ARMS = ["real", "static", "ai"]


def load():
    runs = []
    for group in GROUPS:
        for f in (Path("outputs") / group).glob("*/results.json"):
            d = json.loads(f.read_text())
            d["_group"], d["_path"] = group, str(f)
            d["_seed"] = int(d["config"]["random_state"])
            runs.append(d)
    return runs


def ms(xs):
    return statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0


def fmt(xs):
    m, s = ms(xs)
    return f"{m:.3f} ± {s:.3f}" if len(xs) > 1 else f"{m:.3f}"


def main():
    runs = load()
    lines = [
        "# Resultados FakeWhatsApp.Br — análise consolidada", "",
        f"Gerado em {datetime.now(timezone.utc).isoformat()}. Resultados-fonte são JSONs imutáveis em `outputs/fakewhatsapp-*`.", "",
        "## Protocolo", "",
        "Split temporal: treino jul–ago/2018, validação setembro, teste outubro. Três cenários (full, 10%, 5%), seeds 42/43/44; BERTimbau; prompt WhatsApp v3; Static e AI com teto de 100k tokens/run. Detalhes em `planning/FAKEWHATSAPP.md`.", "",
        "## Resultado geral", "",
        "| Cenário | Braço | Runs | F1 | Precisão | Recall | Tokens | N sintético |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    table = {}
    for group in GROUPS:
        table[group] = {}
        for arm in ARMS:
            rs = sorted((r for r in runs if r["_group"] == group and r["arm"] == arm), key=lambda r: r["_seed"])
            table[group][arm] = rs
            if not rs:
                lines.append(f"| {group} | {arm} | 0 | N/A | N/A | N/A | N/A | N/A |")
                continue
            lines.append(
                f"| {group} | {arm} | {len(rs)} | {fmt([r['test']['f1'] for r in rs])} | "
                f"{fmt([r['test']['precision'] for r in rs])} | {fmt([r['test']['recall'] for r in rs])} | "
                f"{statistics.mean(r.get('tokens_used', 0) for r in rs):.0f} | {statistics.mean(r.get('n_synthetic', 0) for r in rs):.1f} |"
            )

    lines += ["", "## Valores por seed", ""]
    for group in GROUPS:
        lines += [f"### {group}", "", "| Seed | Real | Static | AI |", "|---:|---:|---:|---:|"]
        for seed in (42, 43, 44):
            vals = []
            for arm in ARMS:
                r = next((x for x in table[group][arm] if x["_seed"] == seed), None)
                vals.append(f"{r['test']['f1']:.3f}" if r else "N/A")
            lines.append(f"| {seed} | " + " | ".join(vals) + " |")
        lines.append("")

    lines += ["## Comparações pareadas (exploratórias)", ""]
    for group in GROUPS:
        lines.append(f"### {group}")
        for a, b in (("real", "static"), ("real", "ai"), ("static", "ai")):
            da = {r["_seed"]: r["test"]["f1"] for r in table[group][a]}
            db = {r["_seed"]: r["test"]["f1"] for r in table[group][b]}
            seeds = sorted(set(da) & set(db))
            if len(seeds) >= 2:
                w, p = wilcoxon([da[s] for s in seeds], [db[s] for s in seeds])
                delta = statistics.mean(db[s] - da[s] for s in seeds)
                lines.append(f"- {a} → {b}: ΔF1={delta:+.3f}, Wilcoxon W={w:.1f}, p={p:.3f} (n={len(seeds)}).")
            else:
                lines.append(f"- {a} × {b}: runs insuficientes.")
        lines.append("")

    lines += ["## Qualidade e formato sintético", "", "| Cenário | Braço | Pool sim. | TTR | Wasserstein | Palavras | Recusas | Markdown | Duplicatas | `URGENTE` | `acabei de receber` |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        for arm in ("static", "ai"):
            rs = table[group][arm]
            if not rs:
                continue
            sims, ttrs, wass, words, refusals, markdown, duplicates, urgent, received = [], [], [], [], 0, 0, 0, 0, 0
            for r in rs:
                q = r.get("quality", {})
                sims.append(q.get("diversity", {}).get("mean_pairwise_sim", 0))
                ttrs.append(q.get("coverage", {}).get("ngram_ttr", 0))
                w = q.get("fidelity", {}).get("wasserstein")
                if w is not None: wass.append(w)
                texts = r.get("synthetic", [])
                words.extend(len(t.split()) for t in texts)
                refusal = re.compile(r"(?i)(como (?:um |uma )?(?:modelo|assistente|ia)|não posso (?:ajudar|atender|criar|gerar|fornecer|produzir)|não (?:é possível|posso) (?:cumprir|auxiliar)|desculp[ae][, ]+(?:mas )?não)")
                refusals += sum(bool(refusal.search(t)) for t in texts)
                markdown += sum(t.lstrip().startswith(("#", "**", "```")) for t in texts)
                duplicates += len(texts) - len(set(texts))
                urgent += sum(bool(re.match(r"(?i)^\W*urgente", t)) for t in texts)
                received += sum("acabei de receber" in t.lower() for t in texts)
            total = sum(len(r.get("synthetic", [])) for r in rs)
            lines.append(f"| {group} | {arm} | {statistics.mean(sims):.3f} | {statistics.mean(ttrs):.3f} | {statistics.mean(wass):.3f} | {statistics.mean(words):.1f} | {refusals} | {markdown} | {duplicates} | {urgent/total:.1%} | {received/total:.1%} |")

    lines += ["", "## Síntese automática", ""]
    complete = True
    for group in GROUPS:
        available = {a: table[group][a] for a in ARMS if table[group][a]}
        if any(len(table[group][a]) < 3 for a in ARMS): complete = False
        if available:
            means = {a: statistics.mean(r["test"]["f1"] for r in rs) for a, rs in available.items()}
            best = max(means, key=means.get)
            lines.append(f"- **{group}:** melhor F1 médio = **{best} ({means[best]:.3f})**; " + ", ".join(f"{a}={v:.3f}" for a, v in means.items()) + ".")
    lines += [
        "", "## Como interpretar", "",
        "- Menor `pool_sim` significa maior diversidade semântica; menor Wasserstein significa maior proximidade ao real no proxy.",
        "- O array sintético de AI contém somente o pool herdado, enquanto Static contém tudo gerado; métricas não representam os rejeitados pelo gate.",
        "- n=3 não oferece poder para significância a 5% no Wilcoxon bilateral; p-valores são apenas diagnósticos.",
        "- Comparar eficiência por F1, tokens efetivamente consumidos e N sintético efetivo, não pelo N=500 nominal do Static.",
        "- Mensagens são raw nos três splits; isso corrige o descompasso raw-sintético observado no Fake.br.",
        "- O gerador está preso a um template: no Static, >91% começam por `URGENTE` e >84% contêm `acabei de receber`; nos dados reais esses valores são ~1% e ~0%. AI filtra parte do padrão, mas não o elimina.",
        "", "## Estado", "",
        f"Grade {'COMPLETA' if complete else 'INCOMPLETA'}: {len(runs)}/27 runs encontrados.", "",
        "## Artefatos-fonte", "",
    ]
    lines.extend(f"- `{r['_path']}`" for r in sorted(runs, key=lambda x: (x["_group"], x["arm"], x["_seed"])))

    target = Path("docs/analise_fakewhatsapp.md")
    if target.exists():
        target = Path("docs") / f"analise_fakewhatsapp-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.md"
    target.write_text("\n".join(lines) + "\n")
    print(f"Analysis saved: {target}")


if __name__ == "__main__":
    main()
