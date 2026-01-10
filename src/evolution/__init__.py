"""
Policy Evolution Module

Systematic evolution of context composition policies using objective metrics.
"""

from .evolver import PolicyEvolver, build_policy_variant

__all__ = ["PolicyEvolver", "build_policy_variant"]
