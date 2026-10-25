"""Replay zero-API dos pools sintéticos já salvos (FakeWhatsApp.Br).

Para cada run histórica (Real/Static/AI), treina um BERTimbau NOVO do zero em
D0 ∪ H e avalia no teste temporal, com opção de class weighting e limiar
ajustado na validação. Não chama a API e nunca escreve nos grupos históricos.

Uso:
  uv run python scripts/replay_weighted.py --scenario fakewhatsapp-10pct --class-weight --threshold-tune
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import Config
from src.learner import Learner
from src.metrics import best_threshold, classification_metrics, probabilistic_metrics
from src.runs import save_run
from src.text import normalize_fakebr

SCENARIOS = {
    "fakewhatsapp-full": ("dataset/fakewhatsapp/train-full.csv", "dataset/fakewhatsapp/validation-september.csv", "dataset/fakewhatsapp/test-october.csv"),
    "fakewhatsapp-10pct": ("dataset/fakewhatsapp/train-10pct.csv", "dataset/fakewhatsapp/validation-september.csv", "dataset/fakewhatsapp/test-october.csv"),
    "fakewhatsapp-5pct": ("dataset/fakewhatsapp/train-5pct.csv", "dataset/fakewhatsapp/validation-september.csv", "dataset/fakewhatsapp/test-october.csv"),
    "fakebr-10pct": ("dataset/pre-processed-train-10pct.csv", "dataset/pre-processed-val-10pct.csv", "dataset/pre-processed-test.csv"),
    "fakebr-5pct": ("dataset/pre-processed-train-5pct.csv", "dataset/pre-processed-val-5pct.csv", "dataset/pre-processed-test.csv"),
}
MODEL = "neuralmind/bert-base-portuguese-cased"


def load_csv(path):
    df = pd.read_csv(path)
    texts = df["preprocessed_news"].astype(str).tolist()
    y = (df["label"] == "fake").astype(int).tolist()
    return texts, y


def iter_pools(group: str):
    groups = ([group] if group.startswith("fakewhatsapp") else
              ["e1", "e2", "e3"] if group == "fakebr-10pct" else ["e6-5pct"])
    for name in groups:
      for f in sorted((Path("outputs") / name).glob("*/results.json")):
        d = json.loads(f.read_text())
        cfg = d.get("config", {})
        if group.startswith("fakebr"):
            if d.get("arm") == "static" and d.get("n_synthetic", 0) != 158:
                continue
            if group == "fakebr-5pct" and "5pct" not in cfg.get("csv_path", ""):
                continue
        yield {
            "arm": d.get("arm"),
            "seed": int(cfg.get("random_state", "?")),
            "synthetic": d.get("synthetic", []) or [],
            "n_synthetic": int(d.get("n_synthetic", 0)),
            "tokens_used": int(d.get("tokens_used", 0)),
            "source": str(f),
        }


def evaluate(learner, texts, y, threshold):
    proba = learner.predict(texts)[:, 1]
    thr = threshold if threshold is not None else 0.5
    pred = (proba >= thr).astype(int)
    out = classification_metrics(y, pred)
    out.update(probabilistic_metrics(y, proba))
    out["threshold"] = float(thr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="fakewhatsapp-10pct")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--arm", choices=["real", "static", "ai"], default=None)
    ap.add_argument("--class-weight", dest="class_weight", action="store_true")
    ap.add_argument("--threshold-tune", dest="threshold_tune", action="store_true")
    ap.add_argument("--run-group", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--learner-model", default=MODEL)
    ap.add_argument("--limit", type=int, default=0, help="trunca treino para smoke test (0=desligado)")
    args = ap.parse_args()

    train_csv, val_csv, test_csv = SCENARIOS[args.scenario]
    train_texts, train_y = load_csv(train_csv)
    val_texts, val_y = load_csv(val_csv)
    test_texts, test_y = load_csv(test_csv)
    if args.limit:
        train_texts, train_y = train_texts[: args.limit], train_y[: args.limit]

    run_group = args.run_group or f"{args.scenario}-replay"
    done = 0
    for pool in iter_pools(args.scenario):
        if args.arm and pool["arm"] != args.arm:
            continue
        if args.seed is not None and pool["seed"] != args.seed:
            continue

        cfg = Config(
            run_group=run_group,
            random_state=pool["seed"],
            class_weight=args.class_weight,
            threshold_tune=args.threshold_tune,
            learner_model=args.learner_model,
            device=args.device,
        )

        learner = Learner(
            model_name=args.learner_model,
            random_state=pool["seed"],
            class_weight=args.class_weight,
            device=args.device or None,
        )
        synth = ([normalize_fakebr(s) for s in pool["synthetic"]]
                 if args.scenario.startswith("fakebr") else pool["synthetic"])
        combined_texts = train_texts + synth
        combined_y = train_y + [1] * len(synth)
        learner.fit(combined_texts, combined_y, epochs=cfg.learner_epochs)

        thr = best_threshold(val_y, learner.predict(val_texts)[:, 1]) if args.threshold_tune else None
        test_metrics = evaluate(learner, test_texts, test_y, thr)

        out = {
            "arm": pool["arm"],
            "test": test_metrics,
            "tokens_used": 0,
            "source_tokens_used": pool["tokens_used"],
            "synthetic": synth,
            "n_synthetic": len(synth),
            "scenario": args.scenario,
            "fakebr_normalization": "approximate" if args.scenario.startswith("fakebr") else "none",
            "source": pool["source"],
            "n_fake_train": int(sum(combined_y)),
            "n_true_train": int(len(combined_y) - sum(combined_y)),
        }
        tag = f"{pool['arm']}-replay"
        run_dir = save_run(cfg, out, tag=tag)
        print(f"[{pool['arm']} s{pool['seed']}] f1={test_metrics['f1']:.4f} "
              f"pr_auc={test_metrics['pr_auc']:.4f} thr={test_metrics['threshold']:.2f} "
              f"-> {run_dir}")
        done += 1

    print(f"Done: {done} replays -> outputs/{run_group}/")


if __name__ == "__main__":
    main()
