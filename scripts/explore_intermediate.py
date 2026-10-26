"""Competence sweep + intermediate-band selection (offline, zero API).

Responde "onde a Herança Ativa realmente ajuda?" reusando os candidates.json
zero-shot já gerados (outputs/revision-candidates/):

  (a) competence sweep: varia a fração de dados REAIS usada para treinar o
      learner (curva de competência) e roda score-k vs random-k sobre os
      MESMOS candidatos;
  (b) band-k: herda os k candidatos com score mais próximo da mediana
      (dificuldade intermediária), não os mais difíceis (score-k) nem os
      mais fáceis (easy-k).

Nenhuma chamada de API. Resultados em outputs/<group>/ (imutáveis, resumível:
pula runs cujo results.json já existe).

Usage:
  uv run python scripts/explore_intermediate.py \
    --scenarios covid19br-full,factckbr-full,fakewhatsapp-full \
    --seeds 42 --fractions 0.05,0.10,0.20,0.50,1.00 \
    --arms score-k,random-k --group explore-competence
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.config import Config
from src.data import load_experiment_data
from src.embedder import SemanticEmbedder
from src.evaluator import AdversarialEvaluator
from src.learner import Learner
from src.metrics import best_threshold, classification_metrics, probabilistic_metrics
from src.runs import save_run
from src.text import normalize_fakebr
from scripts.controlled_curatorship import SCENARIOS, paths

CAND_DIR = Path("outputs/revision-candidates")


def find_candidates(scenario: str, seed: int) -> Path:
    hits = sorted(CAND_DIR.glob(f"{scenario}-s{seed}-*/candidates.json"))
    for p in hits:
        meta = json.loads(p.read_text())
        if meta.get("complete") and meta.get("zero_shot"):
            return p
    raise FileNotFoundError(f"No complete zero-shot candidates for {scenario} s{seed}")


def subsample(texts, y, fraction, seed):
    if fraction >= 1.0:
        return texts, y
    _, idx = train_test_split(
        np.arange(len(texts)), test_size=fraction, random_state=seed, stratify=y
    )
    idx = sorted(idx.tolist())
    return [texts[i] for i in idx], [y[i] for i in idx]


def band_indices(scores, k):
    """k índices com score mais próximo da mediana (dificuldade intermediária)."""
    n = len(scores)
    k = min(k, n)
    order = np.argsort(scores)
    start = max(0, n // 2 - k // 2)
    start = min(start, n - k)
    return order[start : start + k].tolist()


def select_indices(arm, candidates, k, rng, evaluator=None, inherited=None):
    k = min(k, len(candidates))
    if arm == "all":
        return list(range(len(candidates)))
    if arm == "random-k":
        return sorted(rng.sample(range(len(candidates)), k))
    scores, _ = evaluator.score(candidates, [1] * len(candidates), inherited or [])
    if arm == "score-k":
        return np.argsort(-scores)[:k].tolist()
    if arm == "easy-k":
        return np.argsort(scores)[:k].tolist()
    if arm == "band-k":
        return band_indices(scores, k)
    raise ValueError(arm)


def run_arm(cfg, train, val, test, batches, arm, fraction, seed, kind):
    tr_texts, tr_y = subsample(train[0], train[1], fraction, seed)
    need_scorer = arm in ("score-k", "easy-k", "band-k")
    embedder = SemanticEmbedder(cfg.semantic_model) if need_scorer else None
    learner = Learner(
        model_name=cfg.learner_model, random_state=seed,
        class_weight=True, device=cfg.device or None,
    )
    learner.fit(tr_texts, tr_y, epochs=cfg.learner_epochs)
    inherited = []
    rng = random.Random(seed)
    evaluator = AdversarialEvaluator(learner, embedder) if need_scorer else None
    for batch in batches:
        candidates = [normalize_fakebr(s) for s in batch] if kind == "fakebr" else batch
        candidates = [s for s in candidates if s.strip()]
        if not candidates:
            raise ValueError("Batch became empty")
        idx = select_indices(arm, candidates, cfg.top_k, rng, evaluator, inherited)
        selected = [candidates[i] for i in idx]
        inherited.extend(selected)
        learner.fit(
            tr_texts + inherited, tr_y + [1] * len(inherited),
            epochs=cfg.learner_epochs,
        )
    learner.fit(
        tr_texts + inherited, tr_y + [1] * len(inherited),
        epochs=cfg.learner_epochs, from_scratch=True,
    )
    val_p = learner.predict(val[0])[:, 1]
    threshold = best_threshold(val[1], val_p)
    p = learner.predict(test[0])[:, 1]
    metrics = classification_metrics(test[1], (p >= threshold).astype(int))
    metrics.update(probabilistic_metrics(test[1], p))
    metrics["threshold"] = threshold
    return metrics, len(inherited)


def already_done(group, seed, arm, fraction):
    tag = f"{arm}-f{fraction}"
    for p in (Path("outputs") / group).glob(f"{tag}-s{seed}-*/results.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if abs(float(d.get("fraction", 1.0)) - fraction) < 1e-9:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="covid19br-full,factckbr-full,fakewhatsapp-full")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--fractions", default="0.05,0.10,0.20,0.50,1.00")
    ap.add_argument("--arms", default="score-k,random-k")
    ap.add_argument("--group", default="explore-competence")
    ap.add_argument("--learner-model", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    fractions = [float(f) for f in args.fractions.split(",") if f.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    for scenario in scenarios:
        csv, val_csv, test_csv, kind = paths(scenario)
        for seed in seeds:
            cand = find_candidates(scenario, seed)
            meta = json.loads(cand.read_text())
            batches = meta["batches"]
            if len(batches) != meta.get("cycles") or any(not b for b in batches):
                raise ValueError(f"Incomplete batches in {cand}")
            cfg = Config(
                csv_path=Path(csv), val_csv_path=Path(val_csv), test_csv_path=Path(test_csv),
                random_state=seed, run_group=args.group, class_weight=True,
                threshold_tune=True, final_from_scratch=True, zero_shot=True,
                learner_model=args.learner_model, device=args.device,
            )
            train, val, test = load_experiment_data(cfg)
            for fraction in fractions:
                for arm in arms:
                    tag = f"{arm}-f{fraction}"
                    if already_done(args.group, seed, arm, fraction):
                        print(f"SKIP {scenario} s{seed} {tag}", flush=True)
                        continue
                    metrics, n_synth = run_arm(cfg, train, val, test, batches, arm, fraction, seed, kind)
                    tr_texts, tr_y = subsample(train[0], train[1], fraction, seed)
                    out = {
                        "arm": arm, "scenario": scenario, "seed": seed,
                        "fraction": fraction, "test": metrics,
                        "n_synthetic": n_synth, "n_real": len(tr_texts),
                        "n_fake_real": int(sum(tr_y)),
                        "source": str(cand),
                    }
                    save_run(cfg, out, tag=tag)
                    print(
                        f"DONE {scenario} s{seed} {arm} f={fraction} "
                        f"F1={metrics['f1']:.4f} rec={metrics['recall']:.4f} "
                        f"n_real={out['n_real']} n_synth={n_synth}",
                        flush=True,
                    )
                    del metrics
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
