"""
Dynamic Context Composer Module

Selects and orders chunks to form a context package under token budget.
"""

from .composer import ContextComposer, render_context_text

__all__ = ["ContextComposer", "render_context_text"]
