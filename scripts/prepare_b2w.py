"""Prepare B2W sentiment (PT reviews) as the 3rd decisive task.

Binarize: overall_rating <= 2 -> negative ("fake", class generated synthetically),
>= 4 -> positive ("true"); rating 3 dropped. Split 60/20/20 (seed 0); D0 = all
positive + 10% negative. Outputs dataset/b2w/{train-10pct,validation,test,
heldout_negative}.csv.

Source: americanas-tech/b2w-reviews01 (main branch).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv"
OUT = Path("dataset/b2w")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(URL, usecols=["review_text", "overall_rating"])
    df = df.dropna(subset=["review_text", "overall_rating"])
    df["label"] = df["overall_rating"].map({1: "fake", 2: "fake", 4: "true", 5: "true"})
    df = df.dropna(subset=["label"])
    df["text"] = df["review_text"].astype(str).str.strip()
    df = df[df.text.str.len() >= 10].drop_duplicates("text").reset_index(drop=True)
    print("negative fraction:", (df.label == "fake").mean().round(3), "n=", len(df))

    tr, rest = train_test_split(df, test_size=0.4, random_state=0, stratify=df.label)
    va, te = train_test_split(rest, test_size=0.5, random_state=0, stratify=rest.label)

    neg = tr[tr.label == "fake"].sample(frac=1, random_state=0)
    d0 = pd.concat([tr[tr.label == "true"], neg.iloc[: int(len(neg) * 0.10)]]).sample(frac=1, random_state=0)
    # subsample for the decisive loop (MiniLM + LR on ~50k would be too slow)
    d0 = pd.concat([d0[d0.label == "fake"], d0[d0.label == "true"].sample(2400, random_state=0)]).sample(frac=1, random_state=0)
    va = pd.concat([va[va.label == "fake"].sample(700, random_state=0), va[va.label == "true"].sample(700, random_state=0)])
    te = pd.concat([te[te.label == "fake"].sample(700, random_state=0), te[te.label == "true"].sample(700, random_state=0)])
    d0_fake = set(d0.text)
    heldout = pd.DataFrame({"text": [t for t in tr[tr.label == "fake"].text if t not in d0_fake]})

    def out(df_):
        return df_.rename(columns={"text": "preprocessed_news"})[["label", "preprocessed_news"]]

    for name, part in [("train-10pct.csv", d0), ("validation.csv", va), ("test.csv", te)]:
        out(part).to_csv(OUT / name, index=False)
    heldout.to_csv(OUT / "heldout_negative.csv", index=False)
    print(f"D0={len(d0)} (negative={sum(d0.label=='fake')}) val={len(va)} test={len(te)} heldout_negative={len(heldout)}")


if __name__ == "__main__":
    main()
