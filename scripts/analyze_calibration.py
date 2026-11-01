"""A2 — calibration/quality analysis of the controlled study (read-only, zero GPU).

Reproduces the advisor's I1 finding: PR-AUC is (almost) identical across the four
inheritance rules, so F1 differences come from the operating point (threshold),
not from a better/worse ranker. Reports per scenario:

  * prior pi of the fake class in validation and test;
  * mean threshold, PR-AUC, F1, MCC and the trivial always-fake F1 = 2*pi/(1+pi);
  * the PR-AUC amplitude (max-min over arms) vs the F1 amplitude.

MCC is derived from precision/recall/accuracy + the test prior (the results.json
does not store the confusion matrix). Oracle-F1 needs test probabilities and is
only reported for runs that saved `test_probs` (new arms).

Usage:
  uv run python scripts/analyze_calibration.py
  uv run python scripts/analyze_calibration.py --latex
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.controlled_curatorship import SCENARIOS as PATHS
from scripts.analyze_normalized import SCENARIOS as DISPLAY, load_group

ARMS = ("all", "random-k", "score-k", "easy-k")


def prior_fake(csv: str) -> float:
    df = pd.read_csv(csv)
    return float((df["label"] == "fake").mean())


def mcc_from(precision, recall, accuracy, pi, n):
    """Confusion matrix from P/R/A + prior; then Matthews correlation."""
    pos = pi * n
    tp = recall * pos
    fp = tp * (1.0 - precision) / precision if precision > 0 else (1.0 - recall) * pos
    fn = pos - tp
    tn = n - tp - fp - fn
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    print(f"{'cenário':<22} {'π_val':>6} {'π_test':>7} {'arm':<9} {'PR-AUC':>8} {'F1':>8} "
          f"{'thr':>6} {'MCC':>7} {'trivF1':>7}")
    rows = []
    for scenario, (prompt, ds, cen) in DISPLAY.items():
        train_csv, val_csv, test_csv, _ = PATHS[scenario]
        pi_val = prior_fake(val_csv)
        pi_test = prior_fake(test_csv)
        n_test = len(pd.read_csv(test_csv))
        group = f"revision-controlled-{scenario}-{prompt}"
        by_arm = {}
        for r in load_group(group):
            if r.get("arm") in ARMS and "f1" in r.get("test", {}):
                by_arm.setdefault(r["arm"], []).append(r["test"])
        arm_pr = {a: [t["pr_auc"] for t in ts] for a, ts in by_arm.items() if ts}
        arm_f1 = {a: [t["f1"] for t in ts] for a, ts in by_arm.items() if ts}
        if not arm_pr:
            continue
        pr_amp = max(mean(v) for v in arm_pr.values()) - min(mean(v) for v in arm_pr.values())
        f1_amp = max(mean(v) for v in arm_f1.values()) - min(mean(v) for v in arm_f1.values())
        for a in ARMS:
            ts = by_arm.get(a, [])
            if not ts:
                continue
            pr = mean([t["pr_auc"] for t in ts])
            f1 = mean([t["f1"] for t in ts])
            thr = mean([t.get("threshold", float("nan")) for t in ts])
            prec = mean([t["precision"] for t in ts])
            rec = mean([t["recall"] for t in ts])
            acc = mean([t["accuracy"] for t in ts])
            mcc = mcc_from(prec, rec, acc, pi_test, n_test)
            triv = 2 * pi_test / (1 + pi_test)
            print(f"{scenario:<22} {pi_val:>6.3f} {pi_test:>7.3f} {a:<9} {pr:>8.4f} {f1:>8.4f} "
                  f"{thr:>6.3f} {mcc:>7.4f} {triv:>7.4f}")
            rows.append((scenario, ds, cen, a, pr, f1, thr, mcc, triv))
        print(f"{'':<22} {'':>6} {'':>7} {'':<9} "
              f"ΔPR={pr_amp:.4f}  ΔF1={f1_amp:.4f}   ← amplitude entre regras")
        print()

    if args.latex:
        print("\n--- LaTeX tab:calibration (por braço, média) ---")
        for scenario, ds, cen, a, pr, f1, thr, mcc, triv in rows:
            print(f"{ds} & {cen} & {a} & {pr:.3f} & {f1:.3f} & {thr:.2f} & {mcc:.3f} & {triv:.3f} \\\\")


if __name__ == "__main__":
    main()
