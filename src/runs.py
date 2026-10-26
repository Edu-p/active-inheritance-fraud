"""Backward-compat shim — movido para src/adapters/runs.py."""
from .adapters.runs import HISTORICAL_GROUPS, experiment_tag, save_run

__all__ = ["HISTORICAL_GROUPS", "experiment_tag", "save_run"]
