"""Loads and splits the pre-processed Fake.br-Corpus (Data 0)."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(cfg):
    df = pd.read_csv(cfg.csv_path)
    df = df[[cfg.label_col, cfg.text_col]].dropna()
    texts = df[cfg.text_col].astype(str).tolist()
    y = (df[cfg.label_col] == cfg.positive_label).astype(int).tolist()
    return texts, y


def split_data(texts, y, cfg):
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    val_frac = cfg.val_size / (1.0 - cfg.test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr,
        test_size=val_frac,
        random_state=cfg.random_state,
        stratify=y_tr,
    )
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)
