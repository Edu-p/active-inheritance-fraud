"""Helpers parametrizados de relatório/LaTeX.

Novos experimentos constroem tabelas via `latex_table(spec)` em vez de
concatenar strings .tex à mão. Agregação de outputs/*/results.json fica aqui.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path("outputs")


def runs(group, arm=None, n=None):
    """Carrega results.json de outputs/<group>/*/results.json (filtros opcionais)."""
    out = []
    for f in (OUT / group).glob("*/results.json"):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if arm and d.get("arm") != arm:
            continue
        cn = d.get("config", {}).get("n_samples")
        if n is not None and str(cn) != str(n):
            continue
        out.append(d)
    return out


def mean_sd(xs):
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, s


def fmt(m_s, nd=3):
    """(media, sd) -> "m ± s" em LaTeX (sem ± se sd == 0)."""
    m, s = m_s
    if s:
        return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"
    return f"{m:.{nd}f}"


def esc(s):
    """Escapa caracteres reservados do LaTeX."""
    return str(s).replace("_", "\\_").replace("%", "\\%")


def latex_table(headers, rows, caption=None, label=None, align=None):
    """Tabela booktabs parametrizada. rows = list[list[str]]."""
    ncol = len(headers)
    a = align or ("l" + "c" * (ncol - 1))
    lines = ["\\begin{table}[t]"]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines += [
        "\\centering",
        f"\\begin{{tabular}}{{{a}}}",
        "\\toprule",
        " & ".join(esc(h) for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)
