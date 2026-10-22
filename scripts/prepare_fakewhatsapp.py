"""Download and create leak-resistant temporal FakeWhatsApp.Br splits.

Train: Jul-Aug 2018; validation: Sep; test: Oct. Scarce trains retain all
non-misinformation and nested 10%/5% samples of misinformation (seed 42).
Never overwrites an existing artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

URL = "https://raw.githubusercontent.com/cabrau/FakeWhatsApp.Br/master/data/2018/fakeWhatsApp.BR_2018_content_only.csv"
SHA256 = "3dc9c80b5c81f85d67bf3e1babf3ac9248db42d6190ea35fe545b4b3d94f2503"
OUT = Path("dataset/fakewhatsapp")


def write_new(df, path):
    if path.exists():
        old = pd.read_csv(path)
        if not old.equals(df.reset_index(drop=True)):
            raise FileExistsError(f"Recuso sobrescrever artefato diferente: {path}")
        return
    with path.open("x") as f:
        df.to_csv(f, index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT / "fakeWhatsApp.BR_2018_content_only.csv"
    if not source.exists():
        data = urlopen(URL, timeout=120).read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("SHA-256 inesperado no download FakeWhatsApp.Br")
        with source.open("xb") as f:
            f.write(data)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Arquivo fonte FakeWhatsApp.Br alterado")

    df = pd.read_csv(source)
    required = {"id", "date", "text", "misinformation"}
    if not required.issubset(df.columns) or len(df) != 6926:
        raise ValueError("Schema/contagem FakeWhatsApp.Br inesperados")
    if df["text"].duplicated().any() or set(df["misinformation"]) != {0, 1}:
        raise ValueError("Esperava textos únicos e rótulos binários")
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y")
    df["label"] = df["misinformation"].map({1: "fake", 0: "true"})
    df = df.rename(columns={"text": "preprocessed_news"})
    cols = ["id", "date", "shares", "label", "preprocessed_news"]

    train = df[df.date < "2018-09-01"].copy()
    val = df[(df.date >= "2018-09-01") & (df.date < "2018-10-01")].copy()
    test = df[df.date >= "2018-10-01"].copy()
    # `id` identifies anonymized users and may recur over time; it is metadata only and
    # never enters the learner. Texts are globally unique, so there is no exact-text leak.
    fake = train[train.label == "fake"].sample(frac=1, random_state=42)
    true = train[train.label == "true"]
    train10 = pd.concat([true, fake.iloc[: int(len(fake) * .10)]]).sample(frac=1, random_state=42)
    train5 = pd.concat([true, fake.iloc[: int(len(fake) * .05)]]).sample(frac=1, random_state=42)

    artifacts = {
        "train-full.csv": train,
        "train-10pct.csv": train10,
        "train-5pct.csv": train5,
        "validation-september.csv": val,
        "test-october.csv": test,
    }
    for name, part in artifacts.items():
        out = part[cols].copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        write_new(out.reset_index(drop=True), OUT / name)

    metadata = {
        "source": URL,
        "source_sha256": SHA256,
        "license": "GPL-3.0",
        "citation": "Cabral et al., FakeWhatsApp.Br, ICEIS 2021",
        "split": "train=2018-07/08; validation=2018-09; test=2018-10",
        "seed": 42,
        "notes": "Text is globally unique. Anonymized user ids may recur across months but are not features.",
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
