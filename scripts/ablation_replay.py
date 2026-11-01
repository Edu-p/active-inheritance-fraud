"""Ablations on the FROZEN candidates: weight sweep + style-blind scorer.

Answers two review critiques without new API generation:

  --mode weights     score-k with different (w_e, w_u, w_d) ratios, to justify
                     the (0.4, 0.4, 0.2) choice. Scorer = BERT learner fit ONCE
                     on real D0 (frozen). The ONLY variable is the weight ratio.
  --mode styleblind  score-k with a SemanticProbe (MiniLM+LR) scorer: content-only,
                     no surface features. Tests whether the score can beat random
                     when the stylistic shortcut is removed.

Protocol (labeled in results): select ALL cycles up-front with a frozen scorer,
then train ONE fresh BERT on D0 ∪ H (from scratch). This isolates the selection
rule; it does NOT replicate the warm-start loop of the main study. Random-k is
the shared control.

Resumable: skips any (scenario, seed, arm) whose results.json already exists.

Usage:
  uv run python scripts/ablation_replay.py --mode weights  --scenarios covid19br-10pct --seeds 42,43,44
  uv run python scripts/ablation_replay.py --mode styleblind --scenarios covid19br-10pct,factckbr-10pct --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.config import Config
from src.data import load_experiment_data
from src.embedder import SemanticEmbedder
from src.evaluator import AdversarialEvaluator
from src.learner import Learner
from src.metrics import best_threshold, classification_metrics, probabilistic_metrics
from src.runs import save_run
from src.text import normalize_fakebr
from src.probe import SemanticProbe
from scripts.controlled_curatorship import SCENARIOS, paths

CAND_DIR = Path("outputs/revision-candidates")

WEIGHT_GRID = {
    "w040_040_020": (0.4, 0.4, 0.2),
    "w100_000_000": (1.0, 0.0, 0.0),   # error only
    "w000_100_000": (0.0, 1.0, 0.0),   # uncertainty only
    "w000_000_100": (0.0, 0.0, 1.0),   # diversity only
    "w050_050_000": (0.5, 0.5, 0.0),   # no diversity
    "w033_033_033": (1 / 3, 1 / 3, 1 / 3),
}


def find_candidates(scenario: str, seed: int) -> Path:
    # ponytail: any complete zero-shot set for the (scenario, seed); the same file
    # is reused across all arms so the comparison stays paired. Matches the
    # existing study's hits[-1] convention (some WhatsApp seeds only have v2).
    hits = sorted(CAND_DIR.glob(f"{scenario}-s{seed}-*/candidates.json"))
    for p in reversed(hits):
        try:
            meta = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if meta.get("complete") and meta.get("zero_shot"):
            return p
    raise FileNotFoundError(f"No canonical candidates for {scenario} s{seed}")


def normalized_batches(batches, kind):
    out = []
    for batch in batches:
        c = [normalize_fakebr(s) for s in batch] if kind == "fakebr" else batch
        c = [s for s in c if s.strip()]
        if c:
            out.append(c)
    return out


def minmax(x):
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    return np.ones_like(x) if span < 1e-12 else (x - x.min()) / span


def select_cycles(batches, weights, scorer_err_unc, all_emb, k, seed, all_candidates):
    """Returns selected indices (flat) using frozen scorer components + diversity."""
    inherited_idx: list[int] = []
    selections: list[list[int]] = []
    flat_offset = 0
    for batch in batches:
        idxs = list(range(flat_offset, flat_offset + len(batch)))
        err, unc = scorer_err_unc(batch)  # per-batch error (log-loss) and uncertainty
        err = minmax(err)
        if inherited_idx:
            P = all_emb[inherited_idx]
            X = all_emb[idxs]
            sim = cosine_similarity(X, P).max(axis=1)
            div = np.clip(1.0 - sim, 0.0, 1.0)
        else:
            div = np.ones(len(idxs))
        scores = weights[0] * err + weights[1] * unc + weights[2] * div
        order = np.argsort(-scores)[: min(k, len(idxs))]
        sel = [idxs[i] for i in order]
        selections.append(sel)
        inherited_idx.extend(sel)
        flat_offset += len(batch)
    return selections, [all_candidates[i] for i in inherited_idx]


def final_fit_eval(cfg, train, val, test, selected_texts, seed):
    learner = Learner(
        model_name=cfg.learner_model, random_state=seed,
        class_weight=True, device=cfg.device or None,
    )
    learner.fit(
        train[0] + selected_texts, train[1] + [1] * len(selected_texts),
        epochs=cfg.learner_epochs, from_scratch=True,
    )
    val_p = learner.predict(val[0])[:, 1]
    threshold = best_threshold(val[1], val_p)
    p = learner.predict(test[0])[:, 1]
    metrics = classification_metrics(test[1], (p >= threshold).astype(int))
    metrics.update(probabilistic_metrics(test[1], p))
    metrics["threshold"] = threshold
    del learner
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def already_done(group, scenario, seed, tag):
    for p in (Path("outputs") / group).glob(f"{tag}-s{seed}-*/results.json"):
        try:
            if json.loads(p.read_text()).get("scenario") == scenario:
                return True
        except (OSError, ValueError):
            continue
    return False


def run_seed(cfg, train, val, test, batches, kind, seed, args):
    embedder = SemanticEmbedder(cfg.semantic_model)
    all_candidates = [t for b in batches for t in b]

    if args.mode == "weights":
        # frozen BERT scorer fit once; error/uncertainty reused across weight ratios
        scorer = Learner(
            model_name=cfg.learner_model, random_state=seed,
            class_weight=True, device=cfg.device or None,
        )
        scorer.fit(train[0], train[1], epochs=cfg.learner_epochs)

        def err_unc(batch):
            e = scorer.error(batch, [1] * len(batch))
            u = scorer.compute_uncertainty(batch)
            return e, u

        arms = [(f"score-{name}", w) for name, w in WEIGHT_GRID.items()]
    else:
        probe = SemanticProbe(embedder)
        probe.fit(train[0], train[1])

        def err_unc(batch):
            e = probe.error(batch, [1] * len(batch))
            u = probe.compute_uncertainty(batch)
            return e, u

        arms = [("score-styleblind", (0.4, 0.4, 0.2))]

    # selection phase (uses frozen scorer; then scorer is freed before final fits)
    all_emb = embedder.transform(all_candidates)
    per_arm: dict[str, tuple[list[str], list[list[int]]]] = {}
    for tag, w in arms:
        selections, selected_texts = select_cycles(
            batches, w, err_unc, all_emb, cfg.top_k, seed, all_candidates
        )
        per_arm[tag] = (selected_texts, selections)

    # random-k shared control (no scorer)
    rng = random.Random(seed)
    rand_sel: list[str] = []
    rand_idx: list[list[int]] = []
    off = 0
    for b in batches:
        idx = sorted(rng.sample(range(len(b)), min(cfg.top_k, len(b))))
        rand_idx.append([off + i for i in idx])
        rand_sel.extend([b[i] for i in idx])
        off += len(b)

    # release frozen scorer/probe before allocating the fresh BERT
    del err_unc
    if args.mode == "weights":
        del scorer
    gc.collect()
    torch.cuda.empty_cache()

    results = [("random-k", rand_sel, rand_idx)]
    for tag, (sel_texts, sel_idx) in per_arm.items():
        results.append((tag, sel_texts, sel_idx))

    for tag, sel_texts, sel_idx in results:
        if already_done(args.group, args.scenario, seed, tag):
            print(f"SKIP {args.scenario} s{seed} {tag}", flush=True)
            continue
        metrics = final_fit_eval(cfg, train, val, test, sel_texts, seed)
        out = {
            "arm": tag, "scenario": args.scenario, "seed": seed,
            "test": metrics, "n_synthetic": len(sel_texts),
            "n_fake_real": int(sum(train[1])),
            "source": str(args.candidates), "selection_indices": sel_idx,
            "protocol": "frozen-scorer, single final from-scratch BERT (no warm-start loop)",
            "mode": args.mode,
        }
        if tag.startswith("score-") and args.mode == "weights":
            name = tag[len("score-"):]
            w = WEIGHT_GRID[name]
            out["weights"] = {"w_error": w[0], "w_uncertainty": w[1], "w_diversity": w[2]}
        save_run(cfg, out, tag=tag)
        print(f"DONE {args.scenario} s{seed} {tag} F1={metrics['f1']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["weights", "styleblind"], required=True)
    ap.add_argument("--scenarios", default="covid19br-10pct")
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--learner-model", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    for scenario in scenarios:
        csv, val_csv, test_csv, kind = paths(scenario)
        for seed in seeds:
            cand = find_candidates(scenario, seed)
            meta = json.loads(cand.read_text())
            batches = normalized_batches(meta["batches"], kind)
            if not batches:
                continue
            group = f"ablation-{args.mode}-{scenario}"
            cfg = Config(
                csv_path=Path(csv), val_csv_path=Path(val_csv), test_csv_path=Path(test_csv),
                random_state=seed, run_group=group, class_weight=True,
                threshold_tune=True, final_from_scratch=True, zero_shot=True,
                learner_model=args.learner_model, device=args.device, top_k=args.top_k,
            )
            train, val, test = load_experiment_data(cfg)
            # attach extra context for run_seed
            args.scenario = scenario
            args.candidates = cand
            args.group = group
            run_seed(cfg, train, val, test, batches, kind, seed, args)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
