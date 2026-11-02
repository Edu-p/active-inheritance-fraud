"""A13 — detectability factorial (advisor's Bloco 2), on the SAME frozen candidates.

Arms (warm-start final, identical policy to the controlled study):

  score-det      score = learner-score - lambda * p_det   (detectability penalty)
  score-frozen   scorer = a BERT frozen on D0 (never saw synthetic)
  realism-k      select the k LEAST detectable candidates (argmin p_det)

p_det = cross-validated TF-IDF+LR real-fake vs candidates (each candidate scored
by a detector that did not see it). Same candidate file per (scenario, seed) as
the controlled study (read from the existing score-k result's "source").

Resumable. Zero API. Saves test_probs/test_preds + pool detectability for checks.

Usage:
  uv run python scripts/shortcut_factorial.py --arms score-det,realism-k --scenarios covid19br-10pct --seeds 42,43
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from src.config import Config
from src.data import load_experiment_data
from src.embedder import SemanticEmbedder
from src.evaluator import AdversarialEvaluator
from src.learner import Learner
from src.metrics import best_threshold, classification_metrics, probabilistic_metrics
from src.runs import save_run
from src.text import normalize_fakebr
from scripts.controlled_curatorship import SCENARIOS, paths
from scripts.analyze_normalized import SCENARIOS as DISPLAY

ARMS = ("score-det", "score-frozen", "realism-k")


def find_candidate_source(scenario, seed):
    """The exact candidates.json used by the controlled study for this cell."""
    prompt = DISPLAY[scenario][0]
    group = Path("outputs") / f"revision-controlled-{scenario}-{prompt}"
    for p in group.glob("score-k-s*/results.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if int(d.get("config", {}).get("random_state", -1)) == seed:
            src = d.get("source")
            if src and Path(src).is_file():
                return Path(src)
    raise FileNotFoundError(f"No controlled score-k run for {scenario} s{seed}")


def normalized_batches(meta, kind):
    out = []
    for batch in meta["batches"]:
        c = [normalize_fakebr(s) for s in batch] if kind == "fakebr" else batch
        c = [s for s in c if s.strip()]
        if c:
            out.append(c)
    return out


def p_detect(candidates, real_fake, seed=42):
    n = len(candidates)
    p = np.zeros(n)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in kf.split(np.arange(n)):
        X = list(real_fake) + [candidates[i] for i in tr]
        y = [0] * len(real_fake) + [1] * len(tr)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(X)
        clf = LogisticRegression(max_iter=2000).fit(vec.transform(X), y)
        p[te] = clf.predict_proba(vec.transform([candidates[i] for i in te]))[:, 1]
    return p


def minmax(x):
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    return np.ones_like(x) if span < 1e-12 else (x - x.min()) / span


def select_by_det(batches, pdet, flat, k):
    """Precompute per-cycle selection by lowest detectability (batch-local indices)."""
    selections = []
    off = 0
    for b in batches:
        idx = list(range(off, off + len(b)))
        order = np.argsort(pdet[idx])[: min(k, len(idx))]
        selections.append(order.tolist())  # batch-local, matches warm_start_final
        off += len(b)
    return selections


def select_by_frozen(batches, scorer, embedder, k):
    """Precompute per-cycle selection with a frozen scorer (error/unc + diversity)."""
    selections = []
    inherited = []
    for b in batches:
        evaluator = AdversarialEvaluator(scorer, embedder)
        s, _ = evaluator.score(b, [1] * len(b), inherited)
        order = np.argsort(-s)[: min(k, len(b))]
        selections.append(order.tolist())
        inherited.extend([b[i] for i in order])
    return selections


def warm_start_final(cfg, train, val, test, batches, selections, seed):
    """Same warm-start loop + final from-scratch BERT as the controlled study."""
    learner = Learner(
        model_name=cfg.learner_model, random_state=seed,
        class_weight=True, device=cfg.device or None,
    )
    learner.fit(train[0], train[1], epochs=cfg.learner_epochs)
    inherited = []
    for batch, idx in zip(batches, selections):
        inherited.extend([batch[i] for i in idx])
        learner.fit(
            train[0] + inherited, train[1] + [1] * len(inherited),
            epochs=cfg.learner_epochs,
        )
    learner.fit(
        train[0] + inherited, train[1] + [1] * len(inherited),
        epochs=cfg.learner_epochs, from_scratch=True,
    )
    val_p = learner.predict(val[0])[:, 1]
    threshold = best_threshold(val[1], val_p)
    p = learner.predict(test[0])[:, 1]
    pred = (p >= threshold).astype(int)
    metrics = classification_metrics(test[1], pred)
    metrics.update(probabilistic_metrics(test[1], p))
    metrics["threshold"] = threshold
    del learner
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, p, pred, len(inherited)


def run_score_det(cfg, train, val, test, batches, all_candidates, pdet, seed, lam):
    learner = Learner(
        model_name=cfg.learner_model, random_state=seed,
        class_weight=True, device=cfg.device or None,
    )
    embedder = SemanticEmbedder(cfg.semantic_model)
    evaluator = AdversarialEvaluator(learner, embedder)
    learner.fit(train[0], train[1], epochs=cfg.learner_epochs)
    inherited = []
    flat = 0
    for b in batches:
        s, _ = evaluator.score(b, [1] * len(b), inherited)
        s = s - lam * pdet[flat:flat + len(b)]
        order = np.argsort(-s)[: min(cfg.top_k, len(b))]
        sel = [b[i] for i in order]
        inherited.extend(sel)
        flat += len(b)
        learner.fit(
            train[0] + inherited, train[1] + [1] * len(inherited),
            epochs=cfg.learner_epochs,
        )
    learner.fit(
        train[0] + inherited, train[1] + [1] * len(inherited),
        epochs=cfg.learner_epochs, from_scratch=True,
    )
    val_p = learner.predict(val[0])[:, 1]
    threshold = best_threshold(val[1], val_p)
    p = learner.predict(test[0])[:, 1]
    pred = (p >= threshold).astype(int)
    metrics = classification_metrics(test[1], pred)
    metrics.update(probabilistic_metrics(test[1], p))
    metrics["threshold"] = threshold
    del learner, embedder
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, p, pred, len(inherited)


def already_done(group, scenario, seed, arm):
    for p in (Path("outputs") / group).glob(f"{arm}-s{seed}-*/results.json"):
        try:
            if json.loads(p.read_text()).get("scenario") == scenario:
                return True
        except (OSError, ValueError):
            continue
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="score-det,score-frozen,realism-k")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--lambda", type=float, default=1.0, dest="lam")
    ap.add_argument("--learner-model", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    for scenario in scenarios:
        if scenario not in SCENARIOS:
            continue
        csv, val_csv, test_csv, kind = paths(scenario)
        for seed in seeds:
            src = find_candidate_source(scenario, seed)
            meta = json.loads(src.read_text())
            batches = normalized_batches(meta, kind)
            if not batches:
                continue
            cfg = Config(
                csv_path=Path(csv), val_csv_path=Path(val_csv), test_csv_path=Path(test_csv),
                random_state=seed, class_weight=True, threshold_tune=True,
                final_from_scratch=True, learner_model=args.learner_model,
                device=args.device,
            )
            train, val, test = load_experiment_data(cfg)
            all_candidates = [t for b in batches for t in b]
            real_fake = [t for t, y in zip(train[0], train[1]) if y == 1]
            pdet = p_detect(all_candidates, real_fake, seed=seed)

            precomputed = {}
            if "score-frozen" in arms:
                scorer = Learner(
                    model_name=args.learner_model, random_state=seed,
                    class_weight=True, device=args.device or None,
                )
                scorer.fit(train[0], train[1], epochs=cfg.learner_epochs)
                embedder = SemanticEmbedder(cfg.semantic_model)
                precomputed["score-frozen"] = select_by_frozen(batches, scorer, embedder, cfg.top_k)
                del scorer, embedder
                gc.collect()
                torch.cuda.empty_cache()
            if "realism-k" in arms:
                precomputed["realism-k"] = select_by_det(batches, pdet, all_candidates, cfg.top_k)

            for arm in arms:
                group = f"shortcut-{arm}-{scenario}"
                if already_done(group, scenario, seed, arm):
                    print(f"SKIP {scenario} s{seed} {arm}", flush=True)
                    continue
                if arm == "score-det":
                    metrics, p, pred, n_synth = run_score_det(
                        cfg, train, val, test, batches, all_candidates, pdet, seed, args.lam
                    )
                    selections = None
                else:
                    selections = precomputed[arm]
                    metrics, p, pred, n_synth = warm_start_final(
                        cfg, train, val, test, batches, selections, seed
                    )
                out = {
                    "arm": arm, "scenario": scenario, "seed": seed,
                    "test": metrics, "n_synthetic": n_synth,
                    "n_fake_real": int(sum(train[1])),
                    "source": str(src), "lambda": args.lam if arm == "score-det" else None,
                    "selection_indices": selections,
                    "test_probs": p.tolist(), "test_preds": pred.tolist(),
                    "pool_detectability_mean": float(pdet.mean()),
                }
                save_run(cfg, out, tag=arm)
                print(f"DONE {scenario} s{seed} {arm} F1={metrics['f1']:.4f} "
                      f"PR-AUC={metrics['pr_auc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
