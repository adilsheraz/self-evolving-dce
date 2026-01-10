"""
Letta Agent Integration for DCE Policy Evolution

This module provides the strategic decision-making layer for autonomous
policy evolution. The Letta agent proposes experiments, analyzes results,
and decides whether to accept or reject policy changes.

Key Components:
- LettaPolicyStrategist: Main interface for policy revisioning
- Memory management for experiment history
- JSON-based proposal and decision formats
"""

from .policy_strategist import LettaPolicyStrategist
from .memory_manager import MemoryBlockManager
from .experiment_tracker import ExperimentTracker

__all__ = [
    "LettaPolicyStrategist",
    "MemoryBlockManager",
    "ExperimentTracker",
]


def create_policy_strategist(
    agent_id: str,
    agent_name: str = "DCE-Policy-Strategist",
    api_key: str = None
) -> LettaPolicyStrategist:
    """
    Factory function to create a LettaPolicyStrategist instance.
    
    Args:
        agent_id: Existing Letta agent ID
        agent_name: Name of the agent (for logging)
        api_key: Letta API key (or uses environment variable)
    
    Returns:
        Configured LettaPolicyStrategist instance
    
    Example:
        >>> letta = create_policy_strategist(
        ...     agent_id="agent-fba21d29-97ff-4726-8a64-fa093b754a02"
        ... )
        >>> proposal = letta.propose_experiment(baseline_metrics, history)
    """
    return LettaPolicyStrategist(
        agent_id=agent_id,
        agent_name=agent_name,
        api_key=api_key
    )
