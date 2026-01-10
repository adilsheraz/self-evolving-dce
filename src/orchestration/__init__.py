"""
Orchestration Module for Self-Evolving DCE System

Combines:
- Letta Policy Strategist (strategic decision-making)
- DCE Engine (deterministic execution)
- Evolution Loop (autonomous improvement)
"""

from .self_evolving_loop import SelfEvolvingLoop

__all__ = ["SelfEvolvingLoop"]
