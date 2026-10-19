"""Output helpers: unique per-run directory, organized by experiment (never overwrites)."""
from __future__ import annotations

import json
from datetime import datetime


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


def save_run(cfg, out, tag="run"):
    exp = experiment_tag(cfg, out.get("arm", tag))
    seed = getattr(cfg, "random_state", "?")
    run_dir = (
        cfg.output_dir / exp / f"{tag}-s{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": {k: str(v) for k, v in vars(cfg).items()}, **out}
    (run_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )
    return run_dir
