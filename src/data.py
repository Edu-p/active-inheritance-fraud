"""Backward-compat shim — movido para src/adapters/data.py."""
from .adapters.data import _texts_labels, load_experiment_data

__all__ = ["_texts_labels", "load_experiment_data"]
