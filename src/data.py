"""Loads the scarce train pool + the balanced holdout (Data 0)."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def _texts_labels(df, cfg):
    df = df[[cfg.label_col, cfg.text_col]].dropna()
    texts = df[cfg.text_col].astype(str).tolist()
    y = (df[cfg.label_col] == cfg.positive_label).astype(int).tolist()
    return texts, y


def load_experiment_data(cfg):
    """train/val = scarce pool (10% fake); test = balanced holdout (intacto)."""
    train_df = pd.read_csv(cfg.csv_path)
    test_df = pd.read_csv(cfg.test_csv_path)

    tr_texts, tr_y = _texts_labels(train_df, cfg)
    te_texts, te_y = _texts_labels(test_df, cfg)

    tr_texts, va_texts, tr_y, va_y = train_test_split(
        tr_texts, tr_y,
        test_size=cfg.val_size,
        random_state=cfg.random_state,
        stratify=tr_y,
    )
    return (tr_texts, tr_y), (va_texts, va_y), (te_texts, te_y)
