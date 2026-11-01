"""Prepare LIAR (short fact-checked statements, English) into the repo schema.

Source: Wang, "Liar, Liar Pants on Fire", ACL 2017.
Download: https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip (train/valid/test TSV).

TSV columns (tab, no header): id, label, statement, subject, speaker, job, state,
party, barely_true_counts, false_counts, half_true_counts, mostly_true_counts,
true_counts, context.

Binarization (strict): {pants-fire, false} -> "fake";
{barely-true, half-true, mostly-true, true} -> "true".
Text = statement (short claim). No date column (LIAR has none usable).

Split: uses LIAR's own train/valid/test. Scarce trains keep all true + nested
10%/5% of fake (seed 42). Never overwrites an existing artifact.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

URL = "https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip"
SHA256 = "611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7"
OUT = Path("dataset/liar")

FAKE = frozenset({"pants-fire", "false"})
TRUE = frozenset({"barely-true", "half-true", "mostly-true", "true"})
COLS = ["label", "preprocessed_news"]


def write_new(df, path):
    if path.exists():
        old = pd.read_csv(path)
        if not old.equals(df.reset_index(drop=True)):
            raise FileExistsError(f"Recuso sobrescrever artefato diferente: {path}")
        return
    with path.open("x") as f:
        df.to_csv(f, index=False)


def parse_tsv(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as f:
        df = pd.read_csv(f, sep="\t", header=None, names=[
            "id", "label", "statement", "subject", "speaker", "job", "state",
            "party", "barely_true", "false", "half_true", "mostly_true",
            "true", "context",
        ])
    lab = df["label"].astype(str).str.strip().str.lower()
    keep = lab.isin(FAKE | TRUE)
    out = df[keep].copy()
    out["label"] = lab[keep].map(lambda v: "fake" if v in FAKE else "true")
    out["preprocessed_news"] = out["statement"].astype(str).str.strip()
    out = out[out.preprocessed_news.astype(bool)]
    out = out.drop_duplicates("preprocessed_news")
    return out[COLS].reset_index(drop=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = urlopen(URL, timeout=180).read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("SHA-256 inesperado no download LIAR")
    zf = zipfile.ZipFile(io.BytesIO(data))
    train_full = parse_tsv(zf, "train.tsv")
    val = parse_tsv(zf, "valid.tsv")
    test = parse_tsv(zf, "test.tsv")

    fake = train_full[train_full.label == "fake"].sample(frac=1, random_state=42)
    true = train_full[train_full.label == "true"]
    train10 = pd.concat([true, fake.iloc[: int(len(fake) * .10)]]).sample(frac=1, random_state=42)
    train5 = pd.concat([true, fake.iloc[: int(len(fake) * .05)]]).sample(frac=1, random_state=42)

    artifacts = {
        "train-full.csv": train_full, "train-10pct.csv": train10,
        "train-5pct.csv": train5, "validation.csv": val, "test.csv": test,
    }
    for name, part in artifacts.items():
        write_new(part.reset_index(drop=True), OUT / name)

    metadata = {
        "source": URL,
        "source_sha256": SHA256,
        "citation": "Wang, 'Liar, Liar Pants on Fire', ACL 2017",
        "split": "LIAR native train/valid/test",
        "binarization": "pants-fire|false -> fake; barely-true|half-true|mostly-true|true -> true",
        "text_col": "statement",
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
