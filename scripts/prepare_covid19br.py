"""Prepare COVID19.BR (WhatsApp misinformation, COVID-19) temporal splits.

Source: Zenodo record 5193932 (ju-resplande/COVID19BR on HF mirrors it).
  - covidbr_labeled.csv : labeled messages (text, misinformation 0/1).
  - covid19br.csv       : full dump with dates (id,date,...text).
Labels are recovered by joining labeled.text -> full.text (earliest date per text).

Temporal split: train = 2020-03 + 2020-04; validation = 2020-06-08..09;
test = 2020-06-10..12 (future months relative to train; no data in May).
Scarce trains keep all non-misinformation and nested 10%/5% samples of
misinformation (seed 42). Never overwrites an existing artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

LABELED_URL = "https://zenodo.org/api/records/5193932/files/covidbr_labeled.csv/content"
FULL_URL = "https://zenodo.org/api/records/5193932/files/covid19br.csv/content"
LABELED_SHA256 = "7a2e1a23fb5d761797a7aa37308e4e09dbc70e2c8bedc688db04a6b54347ba9d"
FULL_SHA256 = "581f29c9ba2b09ae43b742ce3181d163a3e5d7575c5674530db165bc52a604d5"
OUT = Path("dataset/covid19br")


def write_new(df, path):
    if path.exists():
        old = pd.read_csv(path)
        if not old.equals(df.reset_index(drop=True)):
            raise FileExistsError(f"Recuso sobrescrever artefato diferente: {path}")
        return
    with path.open("x") as f:
        df.to_csv(f, index=False)


def fetch(url, sha256, dest):
    if dest.exists():
        data = dest.read_bytes()
    else:
        data = urlopen(url, timeout=300).read()
        with dest.open("xb") as f:
            f.write(data)
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"SHA-256 inesperado: {dest}")
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labeled = OUT / "covidbr_labeled.csv"
    full = OUT / "covid19br.csv"
    fetch(LABELED_URL, LABELED_SHA256, labeled)
    fetch(FULL_URL, FULL_SHA256, full)

    lab = pd.read_csv(labeled)
    ful = pd.read_csv(full)
    required_lab = {"text", "misinformation"}
    if not required_lab.issubset(lab.columns) or lab["text"].duplicated().any():
        raise ValueError("Schema/duplicatas inesperados em covidbr_labeled.csv")
    if not {"text", "date"}.issubset(ful.columns):
        raise ValueError("Schema inesperado em covid19br.csv")

    # Earliest date per text = first appearance of the (possibly viral) message.
    first_date = ful.groupby("text")["date"].min()
    df = lab.merge(first_date.rename("date"), on="text", how="left")
    n_null = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"]).copy()
    df["dt"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
    df = df.dropna(subset=["dt"]).copy()
    df["label"] = df["misinformation"].map({1: "fake", 0: "true"})
    df = df.rename(columns={"text": "preprocessed_news"})
    df = df.drop_duplicates("preprocessed_news").copy()

    train = df[df.dt.dt.month.isin([3, 4])].copy()
    val = df[(df.dt.dt.month == 6) & (df.dt.dt.day.isin([8, 9]))].copy()
    test = df[(df.dt.dt.month == 6) & (df.dt.dt.day.isin([10, 11, 12]))].copy()
    fake = train[train.label == "fake"].sample(frac=1, random_state=42)
    true = train[train.label == "true"]
    train10 = pd.concat([true, fake.iloc[: int(len(fake) * .10)]]).sample(frac=1, random_state=42)
    train5 = pd.concat([true, fake.iloc[: int(len(fake) * .05)]]).sample(frac=1, random_state=42)

    cols = ["date", "shares", "label", "preprocessed_news"]
    artifacts = {
        "train-full.csv": train,
        "train-10pct.csv": train10,
        "train-5pct.csv": train5,
        "validation-june.csv": val,
        "test-june.csv": test,
    }
    for name, part in artifacts.items():
        out = part.copy()
        out["date"] = out["dt"].dt.strftime("%Y-%m-%d")
        write_new(out[cols].reset_index(drop=True), OUT / name)

    metadata = {
        "source": LABELED_URL,
        "source_full": FULL_URL,
        "source_sha256": {"covidbr_labeled.csv": LABELED_SHA256, "covid19br.csv": FULL_SHA256},
        "license": "CC-BY-4.0 (ver Zenodo record 5193932)",
        "citation": "COVID19.BR: A Dataset of Misinformation about COVID-19 in Brazilian Portuguese WhatsApp Messages (DSW/SBBD; Cabral et al.)",
        "split": "train=2020-03/04; validation=2020-06-08..09; test=2020-06-10..12",
        "seed": 42,
        "label_join": "labeled.text -> earliest date in full dump; 14 labeled rows with no date dropped",
        "notes": f"dropped {n_null} rows without a matched date; text globally unique after dedupe.",
        "counts": {
            name: part.label.value_counts().to_dict() | {"total": len(part)}
            for name, part in artifacts.items()
        },
    }
    meta = OUT / "metadata.json"
    encoded = json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    if meta.exists() and meta.read_text() != encoded:
        raise FileExistsError("metadata.json existente é diferente")
    if not meta.exists():
        meta.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
