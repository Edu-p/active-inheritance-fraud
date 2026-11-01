"""N1 + N3: name the real-vs-synthetic style shortcut + measure mode collapse.

Offline, zero API. For each scenario it pools the real `fake` texts and the
frozen zero-shot synthetic candidates (canonical prompt per genre) and reports:

  * univariate separability (AUC) of cheap interpretable surface features;
  * the full TF-IDF+LR shortcut AUC (the reported ~1.000);
  * semantic distance to the real centroid;
  * mode-collapse proxies: mean pairwise embedding similarity + distinct-n.

Results -> outputs/style-gap/<scenario>.json  (immutable, one file per scenario).
Console prints a ranked table of which features carry the shortcut.

Usage:
  uv run python scripts/style_gap.py               # all scenarios
  uv run python scripts/style_gap.py --scenarios covid19br-10pct,factckbr-10pct
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from src.embedder import SemanticEmbedder
from src.text import normalize_fakebr

CAND_DIR = Path("outputs/revision-candidates")
OUT_DIR = Path("outputs/style-gap")

# scenario -> (train csv, canonical prompt, kind)
SCENARIOS = {
    "fakewhatsapp-full": ("dataset/fakewhatsapp/train-full.csv", "v4", "whatsapp"),
    "fakewhatsapp-10pct": ("dataset/fakewhatsapp/train-10pct.csv", "v4", "whatsapp"),
    "fakewhatsapp-5pct": ("dataset/fakewhatsapp/train-5pct.csv", "v4", "whatsapp"),
    "covid19br-full": ("dataset/covid19br/train-full.csv", "v4", "whatsapp"),
    "covid19br-10pct": ("dataset/covid19br/train-10pct.csv", "v4", "whatsapp"),
    "covid19br-5pct": ("dataset/covid19br/train-5pct.csv", "v4", "whatsapp"),
    "factckbr-full": ("dataset/factckbr/train-full.csv", "v2", "news"),
    "factckbr-10pct": ("dataset/factckbr/train-10pct.csv", "v2", "news"),
    "factckbr-5pct": ("dataset/factckbr/train-5pct.csv", "v2", "news"),
    "fakebr-10pct": ("dataset/pre-processed-train-10pct.csv", "v2", "fakebr"),
    "fakebr-5pct": ("dataset/pre-processed-train-5pct.csv", "v2", "fakebr"),
}


def surface_features(text: str) -> dict[str, float]:
    t = text or ""
    n_char = len(t)
    n_tok = len(t.split())
    if n_char == 0 or n_tok == 0:
        return {"n_chars": 0.0, "n_tokens": 0.0, "punct_ratio": 0.0, "upper_ratio": 0.0,
                "digit_ratio": 0.0, "excl_quest_ratio": 0.0, "avg_token_len": 0.0,
                "url_ratio": 0.0, "hashtag_ratio": 0.0}
    return {
        "n_chars": float(n_char),
        "n_tokens": float(n_tok),
        "punct_ratio": len(re.findall(r"[.,;:!?]", t)) / n_char,
        "upper_ratio": sum(c.isupper() for c in t) / n_char,
        "digit_ratio": sum(c.isdigit() for c in t) / n_char,
        "excl_quest_ratio": len(re.findall(r"[!?]", t)) / n_char,
        "avg_token_len": sum(len(w) for w in t.split()) / n_tok,
        "url_ratio": len(re.findall(r"https?://|www\.", t)) / n_tok,
        "hashtag_ratio": t.count("#") / n_tok,
    }


def univariate_auc(real: np.ndarray, synth: np.ndarray) -> float:
    """AUC of a single scalar feature separating real (0) from synthetic (1)."""
    x = np.concatenate([real, synth])
    y = np.concatenate([np.zeros(len(real)), np.ones(len(synth))])
    if np.unique(x).size < 2:
        return 0.5
    try:
        return float(roc_auc_score(y, x))
    except ValueError:
        return 0.5


def cohens_d(real: np.ndarray, synth: np.ndarray) -> float:
    r, s = np.asarray(real, float), np.asarray(synth, float)
    pool = np.sqrt((r.var() + s.var()) / 2)
    return float((s.mean() - r.mean()) / pool) if pool > 1e-12 else 0.0


def find_candidates(scenario: str, prompt: str) -> list[dict]:
    hits = []
    for p in CAND_DIR.glob(f"{scenario}-s*-*/candidates.json"):
        try:
            meta = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if meta.get("complete") and meta.get("zero_shot") and meta.get("prompt") == prompt:
            hits.append(meta)
    return hits


def scenario_data(scenario: str) -> dict:
    csv_path, prompt, kind = SCENARIOS[scenario]
    df = pd.read_csv(csv_path)
    real = [str(x) for x in df.loc[df.label == "fake", "preprocessed_news"].dropna()]
    real = list(dict.fromkeys(real))  # dedup, keep order

    metas = find_candidates(scenario, prompt)
    synth: list[str] = []
    for m in metas:
        for batch in m.get("batches", []):
            synth.extend(batch)
    if kind == "fakebr":
        synth = [normalize_fakebr(s) for s in synth]
    synth = [s for s in (x.strip() for x in synth) if s]
    synth = list(dict.fromkeys(synth))

    cap = 4000
    real, synth = real[:cap], synth[:cap]
    n = min(len(real), len(synth))
    if n < 50:
        raise ValueError(f"{scenario}: only {n} pooled samples; skip")
    # balance classes for the classifier (stratify needs equal groups; subsample larger)
    real_s, synth_s = real[:n], synth[:n]

    # interpretable surface features
    feats = {}
    for name in surface_features(real_s[0]):
        feats[name] = {
            "real_mean": None, "synth_mean": None,
            "auc": None, "cohens_d": None,
        }
    r_feat = {k: np.array([surface_features(t)[k] for t in real_s]) for k in feats}
    s_feat = {k: np.array([surface_features(t)[k] for t in synth_s]) for k in feats}
    for k in feats:
        feats[k]["auc"] = univariate_auc(r_feat[k], s_feat[k])
        feats[k]["cohens_d"] = cohens_d(r_feat[k], s_feat[k])
        feats[k]["real_mean"] = float(r_feat[k].mean())
        feats[k]["synth_mean"] = float(s_feat[k].mean())

    # full shortcut classifier (TF-IDF 1-2 grams + LR)
    x_tr, x_te, y_tr, y_te = train_test_split(
        real_s + synth_s, [0] * n + [1] * n,
        test_size=0.3, random_state=42, stratify=[0] * n + [1] * n,
    )
    clf = make_pipeline(
        TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=20000, sublinear_tf=True),
        LogisticRegression(max_iter=500),
    )
    clf.fit(x_tr, y_tr)
    shortcut_auc = float(roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1]))

    # semantic distance to real centroid + mode collapse
    embedder = SemanticEmbedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    R = embedder.transform(real_s)
    S = embedder.transform(synth_s)
    centroid = R.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12

    def cos_sim_dist(X, c):
        dots = X @ c
        return 1.0 - dots  # 1 - cos-sim to unit centroid

    from sklearn.metrics.pairwise import cosine_similarity

    def mean_pairwise(X):
        if len(X) < 2:
            return 1.0
        sims = cosine_similarity(X)
        iu = np.triu_indices_from(sims, k=1)
        return float(sims[iu].mean())

    def distinct_n(texts, n):
        grams, total = set(), 0
        for t in texts:
            toks = t.split()
            if len(toks) >= n:
                for i in range(len(toks) - n + 1):
                    grams.add(" ".join(toks[i : i + n]))
                    total += 1
        return (len(grams) / total) if total else 0.0

    return {
        "scenario": scenario,
        "prompt": prompt,
        "n_real": len(real_s),
        "n_synth": len(synth_s),
        "n_candidate_sets": len(metas),
        "shortcut_auc": shortcut_auc,
        "features": feats,
        "semantic": {
            "dist_to_real_centroid_real": float(cos_sim_dist(R, centroid).mean()),
            "dist_to_real_centroid_synth": float(cos_sim_dist(S, centroid).mean()),
        },
        "mode_collapse": {
            "mean_pairwise_sim_real": mean_pairwise(R),
            "mean_pairwise_sim_synth": mean_pairwise(S),
            "distinct_1_real": distinct_n(real_s, 1),
            "distinct_1_synth": distinct_n(synth_s, 1),
            "distinct_3_real": distinct_n(real_s, 3),
            "distinct_3_synth": distinct_n(synth_s, 3),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    args = ap.parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        if scenario not in SCENARIOS:
            print(f"SKIP unknown {scenario}")
            continue
        try:
            d = scenario_data(scenario)
        except (ValueError, FileNotFoundError) as e:
            print(f"SKIP {scenario}: {e}")
            continue
        (OUT_DIR / f"{scenario}.json").write_text(
            json.dumps(d, indent=2, ensure_ascii=False)
        )
        feats = sorted(d["features"].items(), key=lambda kv: -abs(kv[1]["auc"] - 0.5))
        print(f"\n=== {scenario} (real={d['n_real']} synth={d['n_synth']}, "
              f"sets={d['n_candidate_sets']}) ===")
        print(f"shortcut_auc (TF-IDF+LR) = {d['shortcut_auc']:.3f}")
        print(f"semantic dist-to-centroid  real={d['semantic']['dist_to_real_centroid_real']:.3f} "
              f"synth={d['semantic']['dist_to_real_centroid_synth']:.3f}")
        print(f"mean_pairwise_sim  real={d['mode_collapse']['mean_pairwise_sim_real']:.3f} "
              f"synth={d['mode_collapse']['mean_pairwise_sim_synth']:.3f}")
        print(f"distinct_3  real={d['mode_collapse']['distinct_3_real']:.3f} "
              f"synth={d['mode_collapse']['distinct_3_synth']:.3f}")
        print(f"{'feature':<18} {'AUC(sep)':>9} {'d':>8} {'real':>10} {'synth':>10}")
        for name, f in feats:
            auc = f["auc"] if f["auc"] is not None else 0.5
            print(f"{name:<18} {auc:>9.3f} {f['cohens_d']:>8.2f} "
                  f"{f['real_mean']:>10.3f} {f['synth_mean']:>10.3f}")


if __name__ == "__main__":
    main()
