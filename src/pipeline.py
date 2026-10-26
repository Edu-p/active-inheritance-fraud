"""Backward-compat shim — movido para src/application/pipeline.py."""
from .application.pipeline import ActiveInheritancePipeline

__all__ = ["ActiveInheritancePipeline"]
