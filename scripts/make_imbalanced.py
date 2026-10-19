"""Gera, uma única vez, os dados do cenário de escassez.

- treino/val: 100% dos `true` + FRAC dos `fake` (default 10%).
- teste: holdout estratificado de TEST_FRAC do dataset ORIGINAL (balanceado, intacto).

Idempotente: se os arquivos já existem, pula (use --force para regenerar).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("dataset/pre-processed.csv"))
    ap.add_argument(
        "--out-train", type=Path,
        default=Path("dataset/pre-processed-imbalanced.csv"),
    )
    ap.add_argument(
        "--out-test", type=Path,
        default=Path("dataset/pre-processed-test.csv"),
    )
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out_train.exists() and args.out_test.exists() and not args.force:
        print("Datasets já existem — nada a fazer (use --force para regenerar).")
        return

    df = pd.read_csv(args.csv)[["label", "preprocessed_news"]].dropna()

    # Holdout de teste: estratificado sobre o dataset ORIGINAL (balanceado).
    rest, test = train_test_split(
        df, test_size=args.test_frac, random_state=args.seed, stratify=df["label"]
    )

    # Treino escasso: 100% dos true + FRAC dos fake.
    fake = rest[rest["label"] == "fake"]
    true = rest[rest["label"] == "true"]
    n_keep = max(1, int(len(fake) * args.frac))
    fake_kept = fake.sample(n=n_keep, random_state=args.seed)
    train = pd.concat([true, fake_kept]).sample(frac=1.0, random_state=args.seed)

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    args.out_test.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.out_train, index=False)
    test.to_csv(args.out_test, index=False)

    print(
        f"train (escasso)  -> {args.out_train}: {len(train)} rows "
        f"(fake={len(fake_kept)}, true={len(true)})"
    )
    print(
        f"test  (original) -> {args.out_test}: {len(test)} rows "
        f"(fake={(test['label'] == 'fake').sum()}, "
        f"true={(test['label'] == 'true').sum()})"
    )


if __name__ == "__main__":
    main()
