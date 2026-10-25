"""Post-processing for the 2 new datasets: plots, markdown analysis, LaTeX + PDF.

Runs after run_new_datasets_night.py completes Phase C. Never overwrites an
existing .tex/.pdf: writes a new timestamped file and compiles it with pdflatex.

Usage: uv run python scripts/finalize_new_datasets.py
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

UV = ["uv", "run", "python"]
ARMS = ("all", "random-k", "score-k", "easy-k")
SCENARIOS = {
    "covid19br-full": ("v4", "COVID19.BR", "full"),
    "covid19br-10pct": ("v4", "COVID19.BR", "10\\%"),
    "covid19br-5pct": ("v4", "COVID19.BR", "5\\%"),
    "factckbr-full": ("v2", "FACTCK.BR", "full"),
    "factckbr-10pct": ("v2", "FACTCK.BR", "10\\%"),
    "factckbr-5pct": ("v2", "FACTCK.BR", "5\\%"),
}
DATASET_LABEL = {
    "covid19br": "COVID19.BR",
    "factckbr": "FACTCK.BR",
}


def run(cmd):
    print("CMD " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


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
    if s:
        return f"{m:.3f} $\\pm$ {s:.3f}"
    return f"{m:.3f}"


def collect():
    data = {}
    for scenario, (prompt, _, _) in SCENARIOS.items():
        group = f"revision-controlled-{scenario}-{prompt}"
        data[scenario] = {}
        for arm in ARMS:
            rs = [r for r in runs(group) if r.get("arm") == arm]
            if not rs:
                continue
            f1 = [r["test"]["f1"] for r in rs]
            rec = [r["test"]["recall"] for r in rs]
            pr = [r["test"]["pr_auc"] for r in rs]
            th = [r["test"]["threshold"] for r in rs]
            ns = [r.get("n_synthetic", 0) for r in rs]
            data[scenario][arm] = {
                "f1": mean_sd(f1), "recall": mean_sd(rec), "pr_auc": mean_sd(pr),
                "thr": mean_sd(th), "n_synth": int(statistics.mean(ns)),
                "n": len(rs),
            }
    return data


def esc(s):
    return (str(s).replace("_", "\\_").replace("%", "\\%"))


def build_tex(data, ts, figs):
    L = []
    add = L.append
    add("% =====================================================================")
    add("% Extensão: 2 novos datasets (COVID19.BR e FACTCK.BR).")
    add(f"% Arquivo novo timestamped ({ts}); NÃO sobrescreve os .tex/.pdf existentes.")
    add("% Metodologia idêntica à revisão 21/out v2 (estudo controlado de curadoria).")
    add("% =====================================================================")
    add("\\documentclass[conference]{IEEEtran}")
    add("\\usepackage[utf8]{inputenc}")
    add("\\usepackage[T1]{fontenc}")
    add("\\usepackage{cite}")
    add("\\usepackage{amsmath,amssymb,amsfonts}")
    add("\\usepackage{graphicx}")
    add("\\usepackage{textcomp}")
    add("\\usepackage{xcolor}")
    add("\\usepackage{booktabs}")
    add("\\usepackage[hidelinks]{hyperref}")
    add("\\usepackage{url}")
    add("\\usepackage{balance}")
    add("\\graphicspath{{./images/}{../figures/revision/}{./}}")
    add("\\setlength{\\textfloatsep}{6pt plus 1pt minus 1pt}")
    add("\\setlength{\\intextsep}{5pt plus 1pt minus 1pt}")
    add("\\setlength{\\floatsep}{5pt plus 1pt minus 1pt}")
    add("\\widowpenalty=10000")
    add("\\clubpenalty=10000")
    add("")
    add("\\begin{document}")
    add("")
    add("\\title{Gera\\c{c}\\~ao Sint\\'etica Guiada por Heran\\c{c}a Ativa: Extens\\~ao Multi-dom\\'inio (COVID19.BR e FACTCK.BR)}")
    add("")
    add("\\author{\\IEEEauthorblockN{Eduardo Alexandre de Amorim\\textsuperscript{1}, Cleber Zanchettin\\textsuperscript{1}}")
    add("\\IEEEauthorblockA{\\textsuperscript{1}Centro de Inform\\'atica, Universidade Federal de Pernambuco, Recife, PE, Brasil\\\\")
    add("\\{eaa5, cz\\}@cin.ufpe.br}}")
    add("")
    add("\\maketitle")
    add("")
    add("\\begin{abstract}")
    add("Esta extens\\~ao replica o estudo controlado de curadoria da revis\\~ao 21/out em dois novos "
        "dom\\'inios do portugu\\^es: COVID19.BR (mensagens de WhatsApp sobre a pandemia, split temporal) e "
        "FACTCK.BR (claims verificados por ag\\^encias de checagem, split aleat\\'orio estratificado). Dados os "
        "mesmos candidatos zero-shot, quatro regras de heran\\c{c}a s\\~ao comparadas---herdar tudo (\\emph{All}), "
        "uma amostra aleat\\'oria (\\emph{Random-}k), o score de erro/incerteza/diversidade (\\emph{Score-}k) e o "
        "gate invertido (\\emph{Easy-}k)---em tr\\^es cen\\'arios de escassez por dom\\'inio (full, 10\\%, 5\\%). "
        "Todos os bra\\c{c}os usam pondera\\c{c}\\~ao de classes, limiar ajustado na valida\\c{c}\\~ao e treino final "
        "do zero. Os resultados (tr\\^es seeds, explorat\\'orio) s\\~ao reportados com F1, recall, PR-AUC e custo; "
        "nenhum resultado hist\\'orico foi sobrescrito.")
    add("\\end{abstract}")
    add("")
    add("\\begin{IEEEkeywords}")
    add("data-centric AI, synthetic data, misinformation, controlled study, Portuguese NLP, COVID19.BR, FACTCK.BR")
    add("\\end{IEEEkeywords}")
    add("")
    add("\\section{Introdu\\c{c}\\~ao}")
    add("")
    add("A revis\\~ao 21/out isolou tr\\^es efeitos confundidos na vers\\~ao anterior do \\emph{ActiveInheritance}: "
        "o score de sele\\c{c}\\~ao, o volume gerado e a pol\\'itica de treino (limiar fixo $0{,}5$ sem pondera\\c{c}\\~ao). "
        "O achado central foi que, dados os mesmos candidatos, o score \\textbf{n\\~ao} domina a sele\\c{c}\\~ao aleat\\'oria, "
        "e que boa parte do colapso de recall sob escassez era efeito do limiar e do desbalanceamento. "
        "Aquela an\\'alise, por\\'em, usou um \\'unico dom\\'inio novo (FakeWhatsApp.Br). Aqui estendemos o mesmo "
        "protocolo a dois dom\\'inios adicionais para verificar se a conclus\\~ao se mant\\'em fora de WhatsApp eleitoral.")
    add("")
    add("\\section{Datasets}")
    add("")
    add("\\begin{itemize}")
    add("\\item \\textbf{COVID19.BR}~-- mensagens de WhatsApp em PT-BR sobre a pandemia, rotuladas manualmente "
        "(misinformation). Split temporal: treino=mar\\c{c}o--abril/2020, valida\\c{c}\\~ao=08--09/junho, "
        "teste=10--12/junho. Classe \\texttt{fake} minorit\\'aria (\\ensuremath{\\approx}1:2{,}5). Fonte: Zenodo 5193932.")
    add("\\item \\textbf{FACTCK.BR}~-- claims verificados por ag\\^encias de checagem (Aos Fatos, Lupa, Truco). "
        "Texto=\\texttt{claimReviewed} (a alega\\c{c}\\~ao); \\texttt{reviewBody} foi exclu\\'ido porque declara o veredito "
        "(vazaria o r\\'otulo). Binariza\\c{c}\\~ao estrita: falso/falsa\\,$\\to$\\,fake; verdadeiro/verdadeira\\,$\\to$\\,true; "
        "demais r\\'otulos descartados. Split aleat\\'orio estratificado 70/15/15 (seed 42): o split temporal foi evitado "
        "porque os r\\'otulos \\texttt{true} se concentram em 2016--2018. Corpus fake-majorit\\'ario (checagens concentram-se "
        "em claims falsos). Fonte: github.com/jghm-f/FACTCK.BR.")
    add("\\end{itemize}")
    add("")
    add("\\section{Protocolo}")
    add("")
    add("Id\\^entico ao da revis\\~ao 21/out: uma gera\\c{c}\\~ao zero-shot por cen\\'ario/seed (\\texttt{candidates.json}) e "
        "quatro replays offline sobre os \\emph{mesmos} lotes. \\texttt{deepseek-chat}, prompt v4 (WhatsApp anti-template) para "
        "COVID19.BR e prompt v2 (not\\'icia) para FACTCK.BR, teto de 100 mil tokens por gera\\c{c}\\~ao, $T{=}5$, batch 64, "
        "top-$k{=}16$. Learner BERTimbau, 3 \\'epocas por fit, pondera\\c{c}\\~ao de classes e limiar na valida\\c{c}\\~ao; "
        "treino final do zero em $D_0\\cup H$. Seeds 42--44 ($n{=}3$, explorat\\'orio).")
    add("")
    add("\\section{Resultados}")
    add("")
    add("Tabela~\\ref{tab:f1} reporta F1 (m\\'edia\\,$\\pm$\\,desvio) por dom\\'inio, cen\\'ario e bra\\c{c}o. "
        "Tabela~\\ref{tab:aux} traz recall, PR-AUC, limiar e n\\'umero de sint\\'eticos herdados.")
    add("")
    add("\\begin{table}[t]")
    add("\\caption{F1 (teste) por dom\\'inio, cen\\'ario e bra\\c{c}o. $n{=}3$ seeds.}\\label{tab:f1}")
    add("\\centering")
    add("\\begin{tabular}{llcccc}")
    add("\\toprule")
    add("Dom\\'inio & Cen. & All & Random-k & Score-k & Easy-k \\\\")
    add("\\midrule")
    for scenario, (_, ds, cen) in SCENARIOS.items():
        cells = []
        for arm in ARMS:
            m = data[scenario].get(arm)
            cells.append(fmt(*m["f1"]) if m else "--")
        add(f"{esc(ds)} & {cen} & " + " & ".join(cells) + " \\\\")
    add("\\bottomrule")
    add("\\end{tabular}")
    add("\\end{table}")
    add("")
    add("\\begin{table}[t]")
    add("\\caption{Recall, PR-AUC, limiar e sint\\'eticos herdados (m\\'edia).}\\label{tab:aux}")
    add("\\centering")
    add("\\begin{tabular}{llcccc}")
    add("\\toprule")
    add("Dom\\'inio & Cen. & Bra\\c{c}o & Recall & PR-AUC & n\\_synth \\\\")
    add("\\midrule")
    for scenario, (_, ds, cen) in SCENARIOS.items():
        for arm in ARMS:
            m = data[scenario].get(arm)
            if not m:
                continue
            add(f"{esc(ds)} & {cen} & {arm} & {m['recall'][0]:.3f} & {m['pr_auc'][0]:.3f} & {m['n_synth']} \\\\")
    add("\\bottomrule")
    add("\\end{tabular}")
    add("\\end{table}")
    add("")
    if figs:
        add("\\section{Figuras}")
        add("")
        add("Fronteiras Pareto de F1\\,$\\times$\\,tokens e F1\\,$\\times$\\,n\\_sint\\'eticos para os cen\\'arios completos:")
        add("\\begin{figure}[t]")
        add("\\centering")
        for f in figs[:2]:
            add("\\includegraphics[width=\\columnwidth]{" + Path(f).name + "}")
        add("\\caption{Pareto F1\\,$\\times$\\,recursos (cen\\'ario full).}\\label{fig:pareto}")
        add("\\end{figure}")
        add("")
        add("Figuras adicionais (todos os cen\\'arios):")
        add("\\begin{itemize}")
        for f in figs:
            add("\\item \\texttt{" + esc(Path(f).name) + "}")
        add("\\end{itemize}")
    add("")
    add("\\section{Limita\\c{c}\\~oes}")
    add("")
    add("\\begin{itemize}")
    add("\\item FACTCK.BR usa split aleat\\'orio (n\\~ao temporal) e prior invertido (fake majorit\\'aria); "
        "n\\~ao comparar diretamente com FakeWhatsApp.Br/COVID19.BR sem ressalvar.")
    add("\\item FACTCK.BR \\'e pequeno (\\ensuremath{\\approx}1.0k claims ap\\'os binariza\\c{c}\\~ao); a classe \\texttt{true} "
        "no teste \\'e escassa (\\ensuremath{\\approx}18 exemplos).")
    add("\\item COVID19.BR n\\~ao tem dados em maio/2020; o teste \\'e um futuro curto (junho).")
    add("\\item $n{=}3$ seeds; n\\~ao declarar signific\\^ancia estat\\'istica.")
    add("\\item R\\'otulos sint\\'eticos s\\~ao assumidos \\texttt{fake} sem checagem humana; veracidade exige evid\\^encia externa.")
    add("\\end{itemize}")
    add("")
    add("\\balance")
    add("\\end{document}")
    return "\n".join(L)


def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # 1. Figures (timestamped PNGs).
    for scenario in SCENARIOS:
        run(UV + ["scripts/diagnose_and_plot.py",
                  "--group", f"revision-controlled-{scenario}-{SCENARIOS[scenario][0]}",
                  "--scenario", scenario])
    # 2. Markdown analysis.
    run(UV + ["scripts/analyze_new_datasets.py"])
    # 3. LaTeX + PDF.
    data = collect()
    figs = sorted(Path("figures/revision").glob("*.png"))
    tex = build_tex(data, ts, figs)
    out_tex = Path(f"report-latex/active-inheritance-new-datasets-{ts}.tex")
    out_tex.write_text(tex)
    print(f"Escrito: {out_tex}")
    for _ in range(3):
        rc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory",
             str(out_tex.parent), str(out_tex.name)],
            capture_output=True, text=True)
        if rc.returncode != 0:
            print("pdflatex failed:\n", rc.stdout[-3000:], rc.stderr[-2000:])
            break
    pdf = out_tex.with_suffix(".pdf")
    print(f"PDF: {pdf} exists={pdf.is_file()}")


if __name__ == "__main__":
    main()
