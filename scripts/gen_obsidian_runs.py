#!/usr/bin/env python3
"""Gera notas Obsidian por-run + índice automático a partir dos results.json.

Fonte única de verdade = outputs/*/results.json (imutáveis).
Uso: uv run python scripts/gen_obsidian_runs.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
VAULT = ROOT / "**__active_obsidian"
RUNS_DIR = VAULT / "runs"
INDEX = VAULT / "Índice de Runs.md"

# grupo -> nota de experimento no vault
EXP_NOTE = {
    "e1": "E1 - Real (10%)",
    "e2": "E2 - Static (10%)",
    "e3": "E3 - ActiveInheritance (10%)",
    "e4": "E4 - Pesos do score",
    "e5": "E5 - Top-k",
    "e6-5pct": "E6 - Fake.br 5%",
    "e7-probe": "E7 - Probe + prompt v2",
    "e8-generators": "E8 - DeepSeek vs GPT",
    "e9-laya": "E9 - Laya zero-shot",
    "e10-laya-gate": "E10 - Laya gate",
    "fakewhatsapp-full": "FakeWhatsApp (resultados)",
    "fakewhatsapp-10pct": "FakeWhatsApp (resultados)",
    "fakewhatsapp-5pct": "FakeWhatsApp (resultados)",
}

ORDER = [
    "e1", "e2", "e3", "e4", "e5", "e6-5pct", "e7-probe", "e8-generators",
    "e9-laya", "e10-laya-gate", "fakewhatsapp-full", "fakewhatsapp-10pct",
    "fakewhatsapp-5pct", "sandbox",
]


def parse_dir_name(name: str) -> tuple[str, str]:
    """'real-s42-2026...' -> ('real', '42'); 'static-n158-s43-...' -> ('static-n158','43')."""
    m = re.match(r"^(.*?)-s(\d+)-", name)
    if not m:
        return name, "?"
    return m.group(1), m.group(2)


def fnum(x) -> str:
    return "—" if x is None else f"{float(x):.4f}"


def quality_parts(q: dict | None) -> tuple[str, str, str]:
    if not q:
        return "—", "—", "—"
    div = q.get("diversity") or {}
    cov = q.get("coverage") or {}
    fid = q.get("fidelity") or {}
    pool_sim = div.get("mean_pairwise_sim")
    ttr = cov.get("ngram_ttr")
    wass = fid.get("wasserstein")
    return (
        "—" if pool_sim is None else f"{float(pool_sim):.4f}",
        "—" if ttr is None else f"{float(ttr):.4f}",
        "—" if wass is None else f"{float(wass):.4f}",
    )


def build_run_note(group: str, run_dir: Path, data: dict, name: str) -> str:
    tag, seed = parse_dir_name(run_dir.name)
    arm = data.get("arm") or tag
    test = data.get("test") or {}
    cfg = data.get("config") or {}
    pool_sim, ttr, wass = quality_parts(data.get("quality"))

    lines = [f"# {group} · {arm} · s{seed}", ""]
    lines.append(f"- **Grupo:** `{group}` · **Braço:** `{arm}` · **Seed:** {seed}")
    lines.append(f"- **Diretório:** `outputs/{group}/{run_dir.name}`")
    lines.append("")

    lines.append("## Teste")
    lines.append("| acurácia | F1 | precisão | recall |")
    lines.append("|---:|---:|---:|---:|")
    lines.append(
        f"| {fnum(test.get('accuracy'))} | {fnum(test.get('f1'))} | "
        f"{fnum(test.get('precision'))} | {fnum(test.get('recall'))} |"
    )
    lines.append("")

    lines.append("## Custo / sintético")
    lines.append(
        f"- tokens_used: {data.get('tokens_used', '—')}\n"
        f"- n_synthetic: {data.get('n_synthetic', '—')}"
    )
    lines.append("")

    lines.append("## Qualidade")
    lines.append("| pool_sim | TTR | Wasserstein |")
    lines.append("|---:|---:|---:|")
    lines.append(f"| {pool_sim} | {ttr} | {wass} |")
    lines.append("")

    lines.append("## Config")
    lines.append(
        f"- modelo: `{cfg.get('openai_model', '—')}` · prompt: `{cfg.get('prompt_version', '—')}`\n"
        f"- epochs: {cfg.get('epochs', '—')} · batch: {cfg.get('batch_size', '—')} · top-k: {cfg.get('top_k', '—')}\n"
        f"- pesos: {cfg.get('w_error', '—')}/{cfg.get('w_uncertainty', '—')}/{cfg.get('w_diversity', '—')}\n"
        f"- learner: `{cfg.get('learner_model', '—')}` · embedder: `{cfg.get('embedder', '—')}`"
    )
    lines.append("")

    history = data.get("history") or []
    if history:
        lines.append("## Histórico por ciclo")
        lines.append("| ciclo | val_f1 | herdados | pool_sim | tokens | err/unc/div |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for h in history:
            val = h.get("val") or {}
            div = (h.get("pool_diversity") or {}).get("mean_pairwise_sim")
            sp = h.get("score_parts") or {}
            lines.append(
                f"| {h.get('epoch')} | {fnum(val.get('f1'))} | {h.get('inherited_total')} | "
                f"{'—' if div is None else f'{float(div):.4f}'} | {h.get('tokens_used')} | "
                f"{sp.get('error', 0):.2f}/{sp.get('uncertainty', 0):.2f}/{sp.get('diversity', 0):.2f} |"
            )
        lines.append("")

    back = EXP_NOTE.get(group, "MOC - Experimentos")
    lines.append(f"← [[Índice de Runs]] · [[{back}]] →")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs_by_group: dict[str, list[tuple[str, Path, dict]]] = {}

    for json_path in sorted(OUTPUTS.rglob("results.json")):
        group = json_path.parent.parent.name
        run_dir = json_path.parent
        data = json.loads(json_path.read_text(encoding="utf-8"))
        runs_by_group.setdefault(group, []).append((run_dir.name, run_dir, data))

    # remove notas antigas para não acumular lixo
    for old in RUNS_DIR.glob("*.md"):
        old.unlink()

    rows: list[str] = []
    for group in ORDER:
        runs = runs_by_group.pop(group, [])
        if not runs:
            continue
        runs.sort(key=lambda r: r[0])
        label = " ⚠️ sandbox (não oficial)" if group == "sandbox" else ""
        rows.append(f"\n## {group}{label} ({len(runs)} runs)")
        rows.append("| Run | Seed | F1 | Recall | Tokens | N sint. |")
        rows.append("|---|---|---:|---:|---:|---:|")
        for name, run_dir, data in runs:
            tag, seed = parse_dir_name(name)
            note_name = f"{group}-{name}"
            test = data.get("test") or {}
            note_path = RUNS_DIR / f"{note_name}.md"
            note_path.write_text(build_run_note(group, run_dir, data, name), encoding="utf-8")
            rows.append(
                f"| [[{note_name}|{tag}-s{seed}]] | {seed} | {fnum(test.get('f1'))} | "
                f"{fnum(test.get('recall'))} | {data.get('tokens_used', '—')} | {data.get('n_synthetic', '—')} |"
            )

    if runs_by_group:
        rows.append("\n## outros")
        for group, runs in sorted(runs_by_group.items()):
            rows.append(f"\n### {group}")
            for name, run_dir, data in runs:
                tag, seed = parse_dir_name(name)
                note_name = f"{group}-{name}"
                (RUNS_DIR / f"{note_name}.md").write_text(
                    build_run_note(group, run_dir, data, name), encoding="utf-8"
                )
                test = data.get("test") or {}
                rows.append(f"- [[{note_name}|{tag}-s{seed}]] — F1 {fnum(test.get('f1'))}")

    header = (
        "# Índice de Runs (auto)\n\n"
        "> Gerado por `scripts/gen_obsidian_runs.py`. Não edite à mão; "
        "regere rode `uv run python scripts/gen_obsidian_runs.py`.\n\n"
        "Cada link abre a nota individual do run em `runs/`.\n"
    )
    total = len(list(RUNS_DIR.glob("*.md")))
    INDEX.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"OK: {total} notas de run geradas — índice em {INDEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
