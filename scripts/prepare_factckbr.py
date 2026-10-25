"""Prepare FACTCK.BR (fact-checking claims, PT-BR) splits.

Source: https://github.com/jghm-f/FACTCK.BR (FACTCKBR.tsv). Columns:
URL, Author, datePublished, claimReviewed, reviewBody, title, ratingValue,
bestRating, alternativeName.

Binary label (strict): alternativeName in {falso,falsa} -> "fake";
{verdadeiro,verdadeira} -> "true"; every other nuance label is dropped so the
positive class is unambiguous. Text = claimReviewed (the claim being checked);
reviewBody is NOT used because it states the verdict and would leak the label.

Split: stratified RANDOM (seed 42) 70/15/15. Temporal split is not used: the
"verdadeiro" claims concentrate in 2016-2018, so a time split would leave the
recent period (2019) without a negative class. Documented, not hidden.

Scarce trains keep all non-fake and nested 10%/5% samples of fake (seed 42).
Never overwrites an existing artifact.

Note: this corpus is fake-majority (fact-checkers check mostly false claims),
so its prior differs from FakeWhatsApp.Br / COVID19.BR.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://raw.githubusercontent.com/jghm-f/FACTCK.BR/master/FACTCKBR.tsv"
SHA256 = "1e90fe8b67af22d0f756a7f831b6a2f66072b5d0d66fb1172f3bda1e748201db"
OUT = Path("dataset/factckbr")

FAKE = frozenset({"falso", "falsa"})
TRUE = frozenset({"verdadeiro", "verdadeira"})


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
    source = OUT / "FACTCKBR.tsv"
    if source.exists():
        data = source.read_bytes()
    else:
        data = urlopen(URL, timeout=120).read()
        with source.open("xb") as f:
            f.write(data)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("SHA-256 inesperado no download FACTCK.BR")

    raw = pd.read_csv(source, sep="\t")
    if not {"claimReviewed", "datePublished", "alternativeName"}.issubset(raw.columns):
        raise ValueError("Schema FACTCK.BR inesperado")
    lab = raw["alternativeName"].astype(str).str.strip().str.lower()
    keep = lab.isin(FAKE | TRUE)
    df = raw[keep].copy()
    df["label"] = lab[keep].map(lambda v: "fake" if v in FAKE else "true")
    df["dt"] = pd.to_datetime(df["datePublished"], errors="coerce")
    # ponytail: date is metadata only (random split); keep rows with missing dates.
    df = df.rename(columns={"claimReviewed": "preprocessed_news"})
    df = df.dropna(subset=["preprocessed_news"]).copy()
    df = df.drop_duplicates("preprocessed_news").copy()
    df = df[df.preprocessed_news.str.strip().astype(bool)].copy()

    # Stratified random split (temporal split would starve the negative class in test).
    rest, test = train_test_split(df, test_size=0.15, random_state=42, stratify=df.label)
    train_full, val = train_test_split(rest, test_size=0.15 / 0.85, random_state=42,
                                       stratify=rest.label)
    fake = train_full[train_full.label == "fake"].sample(frac=1, random_state=42)
    true = train_full[train_full.label == "true"]
    train10 = pd.concat([true, fake.iloc[: int(len(fake) * .10)]]).sample(frac=1, random_state=42)
    train5 = pd.concat([true, fake.iloc[: int(len(fake) * .05)]]).sample(frac=1, random_state=42)

    cols = ["date", "label", "preprocessed_news"]
    artifacts = {
        "train-full.csv": train_full,
        "train-10pct.csv": train10,
        "train-5pct.csv": train5,
        "validation.csv": val,
        "test.csv": test,
    }
    for name, part in artifacts.items():
        out = part.copy()
        out["date"] = out["dt"].apply(lambda v: v.strftime("%Y-%m-%d") if pd.notna(v) else "")
        write_new(out[cols].reset_index(drop=True), OUT / name)

    metadata = {
        "source": URL,
        "source_sha256": SHA256,
        "license": "MIT (ver LICENSE no repositório FACTCK.BR)",
        "citation": "Moreno & Bressan, FACTCK.BR, WebMedia 2019 (doi:10.1145/3323503.3361698)",
        "split": "stratified random 70/15/15 (seed 42); temporal split avoided (verdadeiro absent after 2018)",
        "binarization": "alternativeName in {falso,falsa}->fake; {verdadeiro,verdadeira}->true; demais rótulos descartados",
        "text_col": "claimReviewed (claim under check); reviewBody excluded (would leak verdict)",
        "seed": 42,
        "notes": "fake-majority corpus (claims checked are mostly false); prior differs from FakeWhatsApp.Br/COVID19.BR.",
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
