"""Output helpers: unique per-run directory, organized by experiment (never overwrites)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from uuid import uuid4


def experiment_tag(cfg, arm):
    """Mapeia braço + hyperparams para o subdiretório do experimento (e1..e5, senão sandbox)."""
    default_weights = (
        abs(cfg.w_error - 0.4) < 1e-9
        and abs(cfg.w_uncertainty - 0.4) < 1e-9
        and abs(cfg.w_diversity - 0.2) < 1e-9
    )
    if arm == "real":
        return "e1"
    if arm == "static":
        return "e2"
    if arm == "ai":
        if default_weights and cfg.top_k == 16:
            return "e3"
        if default_weights and cfg.top_k != 16:
            return "e5"
        if not default_weights and cfg.top_k == 16:
            return "e4"
    return "sandbox"


HISTORICAL_GROUPS = {
    "e1", "e2", "e3", "e4", "e5", "e6-5pct", "e7-probe", "e8-generators",
    "e9-laya", "e10-laya-gate", "fakewhatsapp-full", "fakewhatsapp-10pct",
    "fakewhatsapp-5pct",
}


def save_run(cfg, out, tag="run"):
    exp = getattr(cfg, "run_group", "") or "sandbox"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", exp):
        raise ValueError(f"Nome de grupo inválido: {exp!r}")
    if exp in HISTORICAL_GROUPS:
        raise ValueError(f"Grupo histórico protegido: {exp}. Use --run-group novo.")
    seed = getattr(cfg, "random_state", "?")
    run_dir = cfg.output_dir / exp / (
        f"{tag}-s{seed}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
    )
    # ponytail: mkdir exclusivo + UUID; resultados anteriores nunca são sobrescritos.
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {"config": {k: str(v) for k, v in vars(cfg).items()}, **out}
    (run_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )
    return run_dir
