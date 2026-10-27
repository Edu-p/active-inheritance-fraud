"""Agrega TODOS os resultados normalizados (read-only) em tabelas p/ o artigo.

Lê:
  - outputs/revision-controlled-*/   (estudo controlado: all/random-k/score-k/easy-k)
  - outputs/explore-competence/      (varredura de competência)
  - outputs/explore-band/            (faixa intermediária band-k)

Imprime no console e, com --latex, emite também os corpos de tabela booktabs
prontos para colar no artigo. Não treina nada, não chama API.

Usage:
  uv run python scripts/analyze_normalized.py            # console
  uv run python scripts/analyze_normalized.py --latex    # + LaTeX
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

OUT = Path("outputs")

# cenário -> (prompt, rótulo de domínio, rótulo de cenário)
SCENARIOS = {
    "fakewhatsapp-full": ("v4", "FakeWhatsApp.Br", "full"),
    "fakewhatsapp-10pct": ("v4", "FakeWhatsApp.Br", "10\\%"),
    "fakewhatsapp-5pct": ("v4", "FakeWhatsApp.Br", "5\\%"),
    "covid19br-full": ("v4", "COVID19.BR", "full"),
    "covid19br-10pct": ("v4", "COVID19.BR", "10\\%"),
    "covid19br-5pct": ("v4", "COVID19.BR", "5\\%"),
    "factckbr-full": ("v2", "FACTCK.BR", "full"),
    "factckbr-10pct": ("v2", "FACTCK.BR", "10\\%"),
    "factckbr-5pct": ("v2", "FACTCK.BR", "5\\%"),
    "fakebr-10pct": ("v2", "Fake.br", "10\\%"),
    "fakebr-5pct": ("v2", "Fake.br", "5\\%"),
}
ARMS = ("all", "random-k", "score-k", "easy-k")


def ms(xs):
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, s


def load_group(group):
    rows = []
    for p in (OUT / group).glob("*/results.json"):
        try:
            rows.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return rows


def controlled():
    """{scenario: {arm: [(f1, recall, pr_auc), ...]}}"""
    data = {}
    for scenario, (prompt, _, _) in SCENARIOS.items():
        group = f"revision-controlled-{scenario}-{prompt}"
        by_arm = {a: [] for a in ARMS}
        for r in load_group(group):
            if r.get("arm") in by_arm and "f1" in r.get("test", {}):
                by_arm[r["arm"]].append(r)
        data[scenario] = by_arm
    return data


def competence():
    by = {}
    for r in load_group("explore-competence"):
        by.setdefault((r["scenario"], r["fraction"], r["arm"]), []).append(r["test"]["f1"])
    return by


def band():
    by = {}
    for r in load_group("explore-band"):
        by.setdefault((r["scenario"], r["arm"]), []).append(r["test"]["f1"])
    return by


def fmt_ms(pair, nd=3):
    m, s = pair
    if s:
        return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"
    return f"{m:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    # 1) Estudo controlado
    data = controlled()
    print("=== Estudo controlado: F1 (média ± desvio, n seeds) ===")
    header = f"{'Domínio':<16} {'Cen.':<6} {'All':>16} {'Random-k':>16} {'Score-k':>16} {'Easy-k':>16} {'n':>4}"
    print(header)
    latex_f1 = []
    for scenario, (_, ds, cen) in SCENARIOS.items():
        vals = {}
        n = 0
        for arm in ARMS:
            rs = data[scenario][arm]
            n = max(n, len(rs))
            vals[arm] = ms([r["test"]["f1"] for r in rs]) if rs else None
        cells = [f"{vals[a][0]:.3f} ± {vals[a][1]:.3f}" if vals[a] else "--" for a in ARMS]
        print(f"{ds:<16} {cen:<6} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16} {cells[3]:>16} {n:>4}")
        latex_f1.append((ds, cen, vals, n))
    if args.latex:
        print("\n--- LaTeX tab:f1 ---")
        for ds, cen, vals, n in latex_f1:
            if all(v is None for v in vals.values()):
                continue
            cells = [f"{vals[a][0]:.3f} $\\pm$ {vals[a][1]:.3f}" if vals[a] else "--" for a in ARMS]
            print(f"{ds} & {cen} & " + " & ".join(cells) + " \\\\")

    # 2) Competência
    comp = competence()
    if comp:
        print("\n=== Varredura de competência (delta = score − random) ===")
        print(f"{'Domínio':<16} {'frac':>6} {'Random':>8} {'Score':>8} {'Δ':>8}")
        scen = sorted({k[0] for k in comp})
        fracs = sorted({k[1] for k in comp})
        latex_comp = []
        for s in scen:
            for f in fracs:
                rnd = comp.get((s, f, "random-k"), [])
                scr = comp.get((s, f, "score-k"), [])
                if not rnd or not scr:
                    continue
                rm = statistics.mean(rnd)
                sm = statistics.mean(scr)
                ds = next(d for sc, (_, d, _) in SCENARIOS.items() if sc == s)
                print(f"{ds:<16} {f:>6.2f} {rm:>8.3f} {sm:>8.3f} {sm-rm:>+8.3f}")
                latex_comp.append((ds, f, rm, sm, sm - rm))
        if args.latex:
            print("\n--- LaTeX tab:competence ---")
            for ds, f, rm, sm, d in latex_comp:
                sign = "+" if d >= 0 else "-"
                print(f"{ds} & {f:.2f} & {rm:.3f} & {sm:.3f} & ${sign}{abs(d):.3f}$ \\\\")

    # 3) band-k
    bnd = band()
    if bnd:
        print("\n=== band-k (fração 1.00, mean ± sd, n seeds) ===")
        print(f"{'Domínio':<16} {'band-k':>18} {'n':>3}")
        for (s, arm), xs in sorted(bnd.items()):
            ds = next(d for sc, (_, d, _) in SCENARIOS.items() if sc == s)
            m, sd = ms(xs)
            print(f"{ds:<16} {m:.3f} ± {sd:.3f} ({len(xs)})")

    if not data and not comp and not bnd:
        print("Nenhum resultado ainda.")


def ms_from(cell):
    return cell  # placeholder, não usado no fluxo atual


if __name__ == "__main__":
    main()
