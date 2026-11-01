"""Build the RAW Fake.br split for the decisive experiment (v2).

Fake.br-Corpus `full_texts/` keeps the source register (the pre-processed version
hides it). Splits by pair id (fake/N.txt + true/N.txt) 60/20/20, seed 0, like the
advisor's pilot. D0 = all train true + 10% of train fake. The other 90% of train
fakes become the held-out real-fake pool (candidate content for graduated
detectability, with correct labels).

Outputs (dataset/fakebr-raw/):
  train-10pct.csv (D0) · validation.csv · test.csv · heldout_fakes.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CORPUS = Path("dataset/Fake.br-Corpus/full_texts")
OUT = Path("dataset/fakebr-raw")


def read_dir(sub):
    texts, ids = [], []
    for fp in sorted(CORPUS.glob(f"{sub}/*.txt")):
        try:
            t = re.sub(r"\s+", " ", fp.read_text(encoding="utf-8")).strip()
        except (OSError, UnicodeDecodeError):
            continue
        if t:
            texts.append(t)
            ids.append(fp.stem)
    return ids, texts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fake_ids, fake_texts = read_dir("fake")
    true_ids, true_texts = read_dir("true")
    common = sorted(set(fake_ids) & set(true_ids), key=int)
    fi = {i: t for i, t in zip(fake_ids, fake_texts)}
    ti = {i: t for i, t in zip(true_ids, true_texts)}

    # split by pair id
    tr_ids, rest = train_test_split(common, test_size=0.4, random_state=0)
    va_ids, te_ids = train_test_split(rest, test_size=0.5, random_state=0)

    def build(ids, keep_all_fake=True, fake_frac=None):
        rows = []
        for i in ids:
            rows.append({"label": "fake", "preprocessed_news": fi[i]})
            rows.append({"label": "true", "preprocessed_news": ti[i]})
        df = pd.DataFrame(rows)
        if fake_frac is not None:
            f = df[df.label == "fake"].sample(frac=fake_frac, random_state=0)
            df = pd.concat([df[df.label == "true"], f]).sample(frac=1, random_state=0)
        return df.reset_index(drop=True)

    d0 = build(tr_ids, fake_frac=0.10)
    val = build(va_ids)
    test = build(te_ids)
    # held-out real fakes = the 90% of train fakes not in D0
    all_train_fake = [fi[i] for i in tr_ids]
    d0_fake = set(d0[d0.label == "fake"].preprocessed_news)
    heldout = pd.DataFrame({"preprocessed_news": [t for t in all_train_fake if t not in d0_fake]})

    for name, df in [("train-10pct.csv", d0), ("validation.csv", val), ("test.csv", test), ("heldout_fakes.csv", heldout)]:
        df.to_csv(OUT / name, index=False)
    print(f"D0={len(d0)} (fake={sum(d0.label=='fake')}) val={len(val)} test={len(test)} "
          f"heldout_fakes={len(heldout)}")


if __name__ == "__main__":
    main()
