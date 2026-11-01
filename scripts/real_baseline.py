"""A1 — Real arm (zero synthetics) on the controlled-study protocol, n=10.

For each scenario x seed: fresh BERTimbau on D0 only, class weighting, threshold
tuned on validation (identical policy to the controlled study). Saves test
probabilities and predictions so oracle-F1/MCC can be computed later.

Resumable (skips existing results.json). Zero API.

Usage:
  uv run python scripts/real_baseline.py --scenarios covid19br-10pct --seeds 42,43
  bash scripts/run_real_baseline.sh   # all 11 scenarios x 10 seeds
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import Config
from src.data import load_experiment_data
from src.learner import Learner
from src.metrics import best_threshold, classification_metrics, probabilistic_metrics
from src.runs import save_run
from scripts.controlled_curatorship import SCENARIOS, paths


def already_done(group, scenario, seed):
    for p in (Path("outputs") / group).glob(f"real-s{seed}-*/results.json"):
        try:
            if json.loads(p.read_text()).get("scenario") == scenario:
                return True
        except (OSError, ValueError):
            continue
    return False


def run_one(scenario, seed, learner_model, device):
    csv, val_csv, test_csv, _ = paths(scenario)
    group = f"real-baseline-{scenario}"
    cfg = Config(
        csv_path=Path(csv), val_csv_path=Path(val_csv), test_csv_path=Path(test_csv),
        random_state=seed, run_group=group, class_weight=True,
        threshold_tune=True, final_from_scratch=True,
        learner_model=learner_model, device=device,
    )
    train, val, test = load_experiment_data(cfg)
    learner = Learner(
        model_name=learner_model, random_state=seed, class_weight=True,
        device=device or None,
    )
    learner.fit(train[0], train[1], epochs=cfg.learner_epochs, from_scratch=True)
    val_p = learner.predict(val[0])[:, 1]
    threshold = best_threshold(val[1], val_p)
    p = learner.predict(test[0])[:, 1]
    pred = (p >= threshold).astype(int)
    metrics = classification_metrics(test[1], pred)
    metrics.update(probabilistic_metrics(test[1], p))
    metrics["threshold"] = threshold
    out = {
        "arm": "real", "scenario": scenario, "seed": seed, "test": metrics,
        "n_synthetic": 0, "n_fake_real": int(sum(train[1])), "n_real": len(train[0]),
        "test_probs": p.tolist(), "test_preds": pred.tolist(),
    }
    dest = save_run(cfg, out, tag="real")
    del learner
    gc.collect()
    torch.cuda.empty_cache()
    print(f"DONE {scenario} s{seed} real F1={metrics['f1']:.4f} PR-AUC={metrics['pr_auc']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--learner-model", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for scenario in scenarios:
        if scenario not in SCENARIOS:
            continue
        group = f"real-baseline-{scenario}"
        for seed in seeds:
            if already_done(group, scenario, seed):
                print(f"SKIP {scenario} s{seed} real", flush=True)
                continue
            run_one(scenario, seed, args.learner_model, args.device)


if __name__ == "__main__":
    main()
