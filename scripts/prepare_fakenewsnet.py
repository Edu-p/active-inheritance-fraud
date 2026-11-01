"""Prepare FakeNewsNet (PolitiFact subset, English) into the repo schema.

Source: Ahren09/FakeNewsNet on HuggingFace (mirror of Shu et al., 2017).
Zip layout: politifact_fake/<id>/news_article.json, politifact_real/<id>/...,
plus gossipcop_*. Each news_article.json has a "text" field (article body).

Binarization: politifact_fake -> "fake"; politifact_real -> "true".
Text = news_article.text. Split: stratified random 70/15/15 (seed 42).
Scarce trains keep all true + nested 10%/5% of fake (seed 42).

Zip is cached at dataset/fakenewsnet-raw.zip (2 GB); re-downloaded only if absent.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://huggingface.co/datasets/Ahren09/FakeNewsNet/resolve/main/fake_news_data.zip"
SHA256 = "c29bd69d10d7095de1cb8a7c669eaf1388c11c8b2a1cc47979fdbc92b36fc9b5"
OUT = Path("dataset/fakenewsnet")
RAW = Path("dataset/fakenewsnet-raw.zip")
COLS = ["label", "preprocessed_news"]


def write_new(df, path):
    if path.exists():
        old = pd.read_csv(path)
        if not old.equals(df.reset_index(drop=True)):
            raise FileExistsError(f"Recuso sobrescrever artefato diferente: {path}")
        return
    with path.open("x") as f:
        df.to_csv(f, index=False)


def get_raw() -> bytes:
    if RAW.exists():
        return RAW.read_bytes()
    data = urlopen(URL, timeout=600).read()
    RAW.write_bytes(data)
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = get_raw()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("SHA-256 inesperado no FakeNewsNet zip")
    zf = zipfile.ZipFile(RAW)

    rows = []
    for name in zf.namelist():
        if not name.endswith("/news_article.json"):
            continue
        if name.startswith("politifact_fake/"):
            label = "fake"
        elif name.startswith("politifact_real/"):
            label = "true"
        else:
            continue  # gossipcop excluded by default (celebrity gossip, different genre)
        try:
            art = json.loads(zf.read(name))
        except (ValueError, OSError):
            continue
        text = (art.get("text") or "").strip()
        if not text:
            continue
        rows.append({"label": label, "preprocessed_news": text})

    df = pd.DataFrame(rows).drop_duplicates("preprocessed_news")
    df = df[df.preprocessed_news.str.split().str.len() >= 10].reset_index(drop=True)
    if len(df) < 200:
        raise ValueError(f"FakeNewsNet Politifact muito pequeno: {len(df)}")

    rest, test = train_test_split(df, test_size=0.15, random_state=42, stratify=df.label)
    train_full, val = train_test_split(rest, test_size=0.15 / 0.85, random_state=42, stratify=rest.label)
    fake = train_full[train_full.label == "fake"].sample(frac=1, random_state=42)
    true = train_full[train_full.label == "true"]
    train10 = pd.concat([true, fake.iloc[: int(len(fake) * .10)]]).sample(frac=1, random_state=42)
    train5 = pd.concat([true, fake.iloc[: int(len(fake) * .05)]]).sample(frac=1, random_state=42)

    artifacts = {
        "train-full.csv": train_full, "train-10pct.csv": train10,
        "train-5pct.csv": train5, "validation.csv": val, "test.csv": test,
    }
    for name, part in artifacts.items():
        write_new(part[COLS].reset_index(drop=True), OUT / name)

    metadata = {
        "source": URL,
        "source_sha256": SHA256,
        "citation": "Shu et al., FakeNewsNet (2017); mirror Ahren09/FakeNewsNet",
        "split": "stratified random 70/15/15 (seed 42)",
        "binarization": "politifact_fake->fake; politifact_real->true; gossipcop excluded",
        "text_col": "news_article.text",
        "seed": 42,
        "counts": {n: p.label.value_counts().to_dict() | {"total": len(p)} for n, p in artifacts.items()},
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
