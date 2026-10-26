"""Backward-compat shim — movido para src/application/report.py."""
from .application.report import esc, fmt, latex_table, mean_sd, runs

__all__ = ["esc", "fmt", "latex_table", "mean_sd", "runs"]
