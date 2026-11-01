"""A1 proxy — wrong-label fraction per rule via a D0-only learner (CPU, no API).

For each scenario: train a SemanticProbe (MiniLM + LR) on D0 (real only), then
score each arm's inherited synthetic texts. A candidate the D0-learner does NOT
call fake (p(fake) < 0.5) is flagged as a likely wrong presumed label. Reports
the wrong-label fraction per arm — a cheap proxy for the LLM-judge audit.

Usage:
  uv run python scripts/wrong_label_proxy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import Config
from src.data import load_experiment_data
from src.embedder import SemanticEmbedder
from src.probe import SemanticProbe
from scripts.controlled_curatorship import SCENARIOS as PATHS
from scripts.analyze_normalized import SCENARIOS as DISPLAY

ARMS = ("all", "random-k", "score-k", "easy-k")


def load_inherited(scenario):
    prompt = DISPLAY[scenario][0]
    group = Path("outputs") / f"revision-controlled-{scenario}-{prompt}"
    by_arm = {a: [] for a in ARMS}
    for p in group.glob("*/results.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if d.get("arm") in by_arm:
            by_arm[d["arm"]].extend(d.get("synthetic", []))
    return {a: list(dict.fromkeys(t for t in texts if t and t.strip())) for a, texts in by_arm.items()}


def main():
    print(f"{'scenario':<20} " + " ".join(f"{a:>10}" for a in ARMS))
    for scenario, (_, ds, _) in DISPLAY.items():
        csv, val_csv, test_csv, _ = PATHS[scenario]
        cfg = Config(csv_path=Path(csv), val_csv_path=Path(val_csv), test_csv_path=Path(test_csv))
        train, _, _ = load_experiment_data(cfg)
        embedder = SemanticEmbedder(cfg.semantic_model)
        probe = SemanticProbe(embedder)
        probe.fit(train[0], train[1])
        by_arm = load_inherited(scenario)
        fracs = {}
        for a in ARMS:
            texts = by_arm[a]
            if not texts:
                fracs[a] = float("nan")
                continue
            p = probe.predict(texts)[:, 1]
            fracs[a] = float((p < 0.5).mean())
        print(f"{scenario:<20} " + " ".join(f"{fracs[a]:>10.3f}" for a in ARMS))


if __name__ == "__main__":
    main()
