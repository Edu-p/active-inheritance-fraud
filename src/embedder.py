"""Backward-compat shim — movido para src/adapters/embedder.py."""
from .adapters.embedder import SemanticEmbedder, TfidfEmbedder

__all__ = ["SemanticEmbedder", "TfidfEmbedder"]
