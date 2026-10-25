"""Cria validação fixa para o Fake.br a partir dos treinos escassos existentes.

O professor apontou que o Fake.br não tinha split de validação (ablações/escolhas
olhavam o teste). Este script separa 20% estratificado do treino escasso como
validação fixa, sem tocar no holdout de teste.

Idempotente: recusa sobrescrever artefatos diferentes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPECS = [
    ("dataset/pre-processed-imbalanced.csv",      "pre-processed-train-10pct.csv", "pre-processed-val-10pct.csv"),
    ("dataset/pre-processed-imbalanced-5pct.csv", "pre-processed-train-5pct.csv",  "pre-processed-val-5pct.csv"),
]


def write_new(df, path):
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path)
        if not old.equals(df.reset_index(drop=True)):
            raise FileExistsError(f"Recuso sobrescrever artefato diferente: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        df.to_csv(f, index=False)


def main():
    for src, out_train, out_val in SPECS:
        df = pd.read_csv(src)
        train, val = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )
        write_new(train, Path("dataset") / out_train)
        write_new(val, Path("dataset") / out_val)
        print(f"{src} -> train {out_train} ({len(train)}), val {out_val} ({len(val)})")
        print(f"    val: fake={(val['label']=='fake').sum()}, true={(val['label']=='true').sum()}")


if __name__ == "__main__":
    main()
