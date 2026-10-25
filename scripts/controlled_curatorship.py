"""Controlled study: generate shared zero-shot batches, then replay 4 arms offline.

Nothing happens unless --generate or --replay is supplied. --generate spends API
credits; --replay never constructs a generator. Historical outputs stay read-only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import Config
from src.data import load_experiment_data
from src.embedder import SemanticEmbedder
from src.evaluator import AdversarialEvaluator
from src.generator import OpenAIGenerator
from src.learner import Learner
from src.metrics import best_threshold, classification_metrics, probabilistic_metrics
from src.runs import save_run
from src.text import normalize_fakebr

# 4th tuple element = kind: "whatsapp" (prompt v3/v4), "news" (prompt v2),
# "fakebr" (prompt v2 + approximate normalization of the synthetic texts).
SCENARIOS = {
    "fakewhatsapp-full": ("dataset/fakewhatsapp/train-full.csv", "dataset/fakewhatsapp/validation-september.csv", "dataset/fakewhatsapp/test-october.csv", "whatsapp"),
    "fakewhatsapp-10pct": ("dataset/fakewhatsapp/train-10pct.csv", "dataset/fakewhatsapp/validation-september.csv", "dataset/fakewhatsapp/test-october.csv", "whatsapp"),
    "fakewhatsapp-5pct": ("dataset/fakewhatsapp/train-5pct.csv", "dataset/fakewhatsapp/validation-september.csv", "dataset/fakewhatsapp/test-october.csv", "whatsapp"),
    "fakebr-10pct": ("dataset/pre-processed-train-10pct.csv", "dataset/pre-processed-val-10pct.csv", "dataset/pre-processed-test.csv", "fakebr"),
    "fakebr-5pct": ("dataset/pre-processed-train-5pct.csv", "dataset/pre-processed-val-5pct.csv", "dataset/pre-processed-test.csv", "fakebr"),
    "covid19br-full": ("dataset/covid19br/train-full.csv", "dataset/covid19br/validation-june.csv", "dataset/covid19br/test-june.csv", "whatsapp"),
    "covid19br-10pct": ("dataset/covid19br/train-10pct.csv", "dataset/covid19br/validation-june.csv", "dataset/covid19br/test-june.csv", "whatsapp"),
    "covid19br-5pct": ("dataset/covid19br/train-5pct.csv", "dataset/covid19br/validation-june.csv", "dataset/covid19br/test-june.csv", "whatsapp"),
    "factckbr-full": ("dataset/factckbr/train-full.csv", "dataset/factckbr/validation.csv", "dataset/factckbr/test.csv", "news"),
    "factckbr-10pct": ("dataset/factckbr/train-10pct.csv", "dataset/factckbr/validation.csv", "dataset/factckbr/test.csv", "news"),
    "factckbr-5pct": ("dataset/factckbr/train-5pct.csv", "dataset/factckbr/validation.csv", "dataset/factckbr/test.csv", "news"),
}


def paths(scenario):
    csv, val, test, kind = SCENARIOS[scenario]
    for p in (csv, val, test):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{p}: prepare splits first")
    return csv, val, test, kind


def dump(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def generate(args):
    csv, val, test, kind = paths(args.scenario)
    prompt = "v2" if kind in ("fakebr", "news") else args.prompt
    cfg = Config(csv_path=Path(csv), val_csv_path=Path(val), test_csv_path=Path(test),
                 random_state=args.seed, generator="openai", openai_model=args.model,
                 prompt_version=prompt, zero_shot=True,
                 gen_max_tokens=512 if prompt == "v4" else 256,
                 generator_workers=4, max_run_tokens=args.max_run_tokens)
    generator = OpenAIGenerator(cfg)
    base = Path("outputs/revision-candidates")
    run_dir = base / f"{args.scenario}-s{args.seed}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"scenario": args.scenario, "seed": args.seed, "prompt": cfg.prompt_version,
                "model": args.model, "zero_shot": True, "cycles": args.cycles,
                "batch_size": args.batch_size, "max_run_tokens": args.max_run_tokens,
                "batches": [], "complete": False}
    dump(run_dir / "candidates.json", manifest)
    for cycle in range(args.cycles):
        batch = [x for x in generator.generate_batch([], args.batch_size) if x.strip()]
        if not batch:
            break
        manifest["batches"].append(batch)
        manifest.update(tokens_used=generator.tokens_used, prompt_tokens=generator.prompt_tokens,
                        completion_tokens=generator.completion_tokens, cost_usd=generator.cost_usd)
        dump(run_dir / "candidates.json", manifest)  # checkpoints on API failure
        print(f"{run_dir}: cycle {cycle} n={len(batch)} tokens={generator.tokens_used}", flush=True)
    manifest["complete"] = len(manifest["batches"]) == args.cycles
    dump(run_dir / "candidates.json", manifest)
    if not manifest["complete"]:
        raise RuntimeError(f"Incomplete candidate set: {run_dir} (kept; do not compare)")
    print(f"Saved shared candidates: {run_dir / 'candidates.json'}")


def select_indices(arm, candidates, k, rng, evaluator=None, inherited=None):
    k = min(k, len(candidates))
    if arm == "all":
        return list(range(len(candidates)))
    if arm == "random-k":
        return sorted(rng.sample(range(len(candidates)), k))
    scores, _ = evaluator.score(candidates, [1] * len(candidates), inherited or [])
    return (np.argsort(scores)[:k] if arm == "easy-k" else np.argsort(-scores)[:k]).tolist()


def replay(args):
    source = Path(args.candidates)
    meta = json.loads(source.read_text())
    scenario, seed = meta["scenario"], int(meta["seed"])
    if scenario not in SCENARIOS or not meta.get("complete") or not meta.get("zero_shot"):
        raise ValueError("Requires a complete zero-shot candidate set")
    batches = meta["batches"]
    if len(batches) != meta["cycles"] or any(not b for b in batches):
        raise ValueError("Incomplete candidate batches")
    csv, val, test, kind = paths(scenario)
    cfg = Config(csv_path=Path(csv), val_csv_path=Path(val), test_csv_path=Path(test),
                 random_state=seed, run_group=args.run_group, class_weight=True,
                 threshold_tune=True, final_from_scratch=True, zero_shot=True,
                 learner_model=args.learner_model, device=args.device,
                 top_k=args.top_k)
    train, validation, holdout = load_experiment_data(cfg)
    embedder = SemanticEmbedder(cfg.semantic_model) if args.arm in ("score-k", "easy-k") else None
    learner = Learner(model_name=args.learner_model, random_state=seed,
                      class_weight=True, device=args.device or None)
    learner.fit(*train, epochs=cfg.learner_epochs)
    inherited = []
    selections = []
    rng = random.Random(seed)
    evaluator = AdversarialEvaluator(learner, embedder) if embedder else None
    for batch in batches:
        raw = batch
        # ponytail: Fake.br normalization is approximate, not exact historical preprocessing.
        candidates = [normalize_fakebr(s) for s in raw] if kind == "fakebr" else raw
        candidates = [s for s in candidates if s.strip()]
        if not candidates:
            raise ValueError("Batch became empty after normalization")
        indices = select_indices(args.arm, candidates, args.top_k, rng, evaluator, inherited)
        selected = [candidates[i] for i in indices]
        selections.append(indices)
        inherited.extend(selected)
        learner.fit(train[0] + inherited, train[1] + [1] * len(inherited), epochs=cfg.learner_epochs)
    # Exact same final protocol for all arms: independent fresh BERT in D0 ∪ H.
    learner.fit(train[0] + inherited, train[1] + [1] * len(inherited),
                epochs=cfg.learner_epochs, from_scratch=True)
    val_p = learner.predict(validation[0])[:, 1]
    threshold = best_threshold(validation[1], val_p)
    p = learner.predict(holdout[0])[:, 1]
    metrics = classification_metrics(holdout[1], (p >= threshold).astype(int))
    metrics.update(probabilistic_metrics(holdout[1], p))
    metrics["threshold"] = threshold
    out = {"arm": args.arm, "scenario": scenario, "test": metrics,
           "source": str(source), "selection_indices": selections,
           "n_synthetic": len(inherited), "synthetic": inherited,
           "n_fake_train": sum(train[1]) + len(inherited), "n_true_train": len(train[1]) - sum(train[1]),
           "tokens_used": meta.get("tokens_used", 0), "prompt_tokens": meta.get("prompt_tokens", 0),
           "completion_tokens": meta.get("completion_tokens", 0), "cost_usd": meta.get("cost_usd", 0.0),
           "fakebr_normalization": "approximate" if kind == "fakebr" else "none"}
    output = save_run(cfg, out, tag=args.arm)
    print(f"{args.arm} s{seed} F1={metrics['f1']:.4f} -> {output}")


def main():
    ap = argparse.ArgumentParser()
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate", action="store_true", help="calls paid API explicitly")
    action.add_argument("--replay", action="store_true", help="offline BERT replay")
    ap.add_argument("--scenario", choices=SCENARIOS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--prompt", choices=["v2", "v3", "v4"], default="v3")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-run-tokens", type=int, default=100000)
    ap.add_argument("--candidates", type=Path)
    ap.add_argument("--arm", choices=["all", "random-k", "score-k", "easy-k"])
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--run-group", default="revision-controlled")
    ap.add_argument("--learner-model", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    if args.generate:
        if not args.scenario or args.cycles < 1 or args.batch_size < 1:
            ap.error("--generate needs --scenario and positive --cycles/--batch-size")
        generate(args)
    else:
        if not args.candidates or not args.arm or args.top_k < 1:
            ap.error("--replay needs --candidates, --arm and positive --top-k")
        replay(args)


if __name__ == "__main__":
    main()
