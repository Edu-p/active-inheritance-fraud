"""Prepare ToLD-Br (Brazilian Portuguese toxicity) as the 2nd decisive task.

Binarizes: toxic = any dimension (homophobia/obscene/insult/racism/misogyny/
xenophobia) >= 1 vote. Split 60/20/20 (seed 0); D0 = all non-toxic + 10% toxic.
Outputs dataset/toldbr/{train-10pct,validation,test,heldout_toxic}.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://raw.githubusercontent.com/JAugusto97/ToLD-Br/master/ToLD-BR.csv"
OUT = Path("dataset/toldbr")
DIMS = ["homophobia", "obscene", "insult", "racism", "misogyny", "xenophobia"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(URL)
    raw["toxic"] = (raw[DIMS].sum(axis=1) >= 1).astype(int)
    df = raw[["text", "toxic"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df.text.str.len() >= 5].drop_duplicates("text").reset_index(drop=True)
    print("toxic fraction:", df.toxic.mean().round(3), "n=", len(df))

    tr, rest = train_test_split(df, test_size=0.4, random_state=0, stratify=df.toxic)
    va, te = train_test_split(rest, test_size=0.5, random_state=0, stratify=rest.toxic)

    tox = tr[tr.toxic == 1].sample(frac=1, random_state=0)
    d0 = pd.concat([tr[tr.toxic == 0], tox.iloc[: int(len(tox) * 0.10)]]).sample(frac=1, random_state=0)
    d0_fake = set(d0.text)
    heldout = pd.DataFrame({"text": [t for t in tr[tr.toxic == 1].text if t not in d0_fake]})

    def out(df_):
        return df_.rename(columns={"text": "preprocessed_news"}).assign(
            label=df_["toxic"].map({1: "fake", 0: "true"}))[["label", "preprocessed_news"]]

    for name, part in [("train-10pct.csv", d0), ("validation.csv", va), ("test.csv", te)]:
        out(part).to_csv(OUT / name, index=False)
    heldout.to_csv(OUT / "heldout_toxic.csv", index=False)
    print(f"D0={len(d0)} (toxic={sum(d0.toxic)}) val={len(va)} test={len(te)} heldout_toxic={len(heldout)}")


if __name__ == "__main__":
    main()
