"""Backward-compat shim — movido para src/adapters/generator.py."""
from .adapters.generator import BaseGenerator, MockGenerator, OpenAIGenerator

__all__ = ["BaseGenerator", "MockGenerator", "OpenAIGenerator"]
