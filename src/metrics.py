"""Backward-compat shim — movido para src/domain/metrics.py."""
from .domain.metrics import (
    best_threshold,
    classification_metrics,
    coverage_metrics,
    diversity_metrics,
    probabilistic_metrics,
    synthetic_quality,
    wasserstein_fidelity,
)

__all__ = [
    "best_threshold",
    "classification_metrics",
    "coverage_metrics",
    "diversity_metrics",
    "probabilistic_metrics",
    "synthetic_quality",
    "wasserstein_fidelity",
]
