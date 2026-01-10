"""
Letta Policy Strategist - Strategic decision-making for DCE policy evolution.

This module provides the interface between DCE and the Letta agent,
enabling autonomous policy evolution through strategic reasoning.
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class LettaPolicyStrategist:
    """
    Letta agent interface for autonomous policy evolution.
    
    This class wraps a Letta agent and provides methods for:
    - Proposing next policy experiments
    - Analyzing experimental results
    - Making ACCEPT/REJECT decisions
    - Updating agent memory with experiment history
    
    The agent uses memory blocks to maintain:
    - base_policy: Reference baseline policy (v0.1)
    - experiment_log: History of all tested variants
    - performance_metrics: Current evaluation results
    - strategy_notes: Agent's reasoning and learnings
    
    Example:
        >>> strategist = LettaPolicyStrategist(
        ...     agent_id="agent-fba21d29-97ff-4726-8a64-fa093b754a02"
        ... )
        >>> 
        >>> # Propose next experiment
        >>> proposal = strategist.propose_experiment(
        ...     baseline_metrics={"overall": 0.782, "novelty": 0.65},
        ...     experiment_history=[...]
        ... )
        >>> 
        >>> # Execute experiment using DCE
        >>> results = dce_engine.run_experiment(proposal["override"])
        >>> 
        >>> # Let agent decide
        >>> decision = strategist.decide_on_results(proposal, results)
        >>> if decision["decision"] == "ACCEPT":
        ...     dce_engine.update_policy(proposal["override"])
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_name: str = "DCE-Policy-Strategist",
        api_key: Optional[str] = None
    ):
        """
        Initialize Letta Policy Strategist.
        
        Args:
            agent_id: Existing Letta agent ID (e.g., "agent-fba21d29...")
            agent_name: Human-readable name for logging
            api_key: Letta API key (or uses LETTA_API_KEY env var)
        
        Raises:
            ValueError: If API key is not provided and not in environment
            ImportError: If letta-client is not installed
        """
        try:
            from letta_client import Letta
        except ImportError:
            raise ImportError(
                "letta-client not installed. Install with: pip install letta-client"
            )
        
        # Get API key
        self.api_key = api_key or os.getenv("LETTA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "LETTA_API_KEY not found. Provide via parameter or environment variable."
            )
        
        self.agent_id = agent_id
        self.agent_name = agent_name
        
        # Connect to Letta
        self.client = Letta(api_key=self.api_key)
        self.agent = self.client.agents.retrieve(agent_id)
        
        # Verify connection
        if not self.agent:
            raise ValueError(f"Agent not found: {agent_id}")
        
        print(f"✅ Connected to Letta agent: {self.agent_name}")
        print(f"   Agent ID: {self.agent_id}")
    
    def propose_experiment(
        self,
        baseline_metrics: Dict[str, float],
        experiment_history: List[Dict[str, Any]],
        base_policy: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ask Letta agent to propose next policy experiment.
        
        The agent analyzes baseline performance and experiment history
        to strategically suggest the next parameter variation to test.
        
        Args:
            baseline_metrics: Current baseline scores (e.g., {"overall": 0.782})
            experiment_history: List of past experiments with results
            base_policy: Optional baseline policy dict (for reference)
        
        Returns:
            Proposal dict with:
            {
                "version": "v0.6",
                "override": {"w_novelty": 0.35, "w_retrieval": 0.45},
                "rationale": "Baseline shows low novelty. Increasing weight...",
                "acceptance_rule": "delta_overall > 0.01 AND delta_novelty > 0.05"
            }
        
        Example:
            >>> proposal = strategist.propose_experiment(
            ...     baseline_metrics={"overall": 0.782, "novelty": 0.65},
            ...     experiment_history=[{
            ...         "version": "v0.5",
            ...         "override": {"w_novelty": 0.30},
            ...         "delta_overall": 0.008,  # Below threshold
            ...         "decision": "REJECT"
            ...     }]
            ... )
            >>> print(proposal["override"])
            {'w_novelty': 0.35, 'w_retrieval': 0.45}
        """
        # Format message for agent
        history_summary = self._format_history(experiment_history)
        
        message = f"""
        Current Baseline Metrics:
        {json.dumps(baseline_metrics, indent=2)}
        
        Experiment History (last 5):
        {history_summary}
        
        Task: Propose the next controlled policy experiment.
        Format your response as JSON:
        {{
            "version": "v0.X",
            "override": {{"param": value}},
            "rationale": "Why this experiment...",
            "acceptance_rule": "Criteria for accepting..."
        }}
        
        Guidelines:
        - Change ONE parameter at a time (controlled experiment)
        - Be conservative (small incremental changes)
        - Avoid re-testing failed combinations
        - Provide clear acceptance criteria
        """
        
        # Send message to agent
        response = self.client.agents.messages.create(
            agent_id=self.agent_id,
            messages=[{"role": "user", "content": message}]
        )
        
        # Get latest messages
        msgs = list(self.client.agents.messages.list(self.agent_id, limit=10))
        
        # Find latest assistant message
        agent_reply = None
        for m in reversed(msgs):
            if m.message_type == "assistant_message":
                agent_reply = m
                break
        
        if not agent_reply:
            raise ValueError("No assistant response found")
        
        # Parse JSON from response
        proposal = self._extract_json(agent_reply.content)
        
        return proposal
    
    def decide_on_results(
        self,
        proposal: Dict[str, Any],
        results: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Ask Letta agent to decide ACCEPT or REJECT based on results.
        
        The agent evaluates whether the experimental results meet the
        acceptance criteria defined in the proposal.
        
        Args:
            proposal: Original proposal from propose_experiment()
            results: Experimental results (e.g., {"delta_overall": 0.015})
        
        Returns:
            Decision dict with:
            {
                "decision": "ACCEPT" | "REJECT",
                "notes": "Reasoning for decision...",
                "next_proposal": {...}  # If REJECT, suggest next experiment
            }
        
        Example:
            >>> decision = strategist.decide_on_results(
            ...     proposal={"version": "v0.6", "acceptance_rule": "delta > 0.01"},
            ...     results={"delta_overall": 0.015}
            ... )
            >>> print(decision["decision"])
            'ACCEPT'
        """
        message = f"""
        Experiment Results for {proposal['version']}:
        
        Proposal:
        {json.dumps(proposal, indent=2)}
        
        Actual Results:
        {json.dumps(results, indent=2)}
        
        Task: Decide ACCEPT or REJECT based on acceptance rule.
        Format response as JSON:
        {{
            "decision": "ACCEPT" or "REJECT",
            "notes": "Reasoning...",
            "next_proposal": {{...}}  # Only if REJECT
        }}
        """
        
        response = self.client.agents.messages.create(
            agent_id=self.agent_id,
            messages=[{"role": "user", "content": message}]
        )
        
        # Get latest messages
        msgs = list(self.client.agents.messages.list(self.agent_id, limit=10))
        
        # Find latest assistant message
        agent_reply = None
        for m in reversed(msgs):
            if m.message_type == "assistant_message":
                agent_reply = m
                break
        
        if not agent_reply:
            raise ValueError("No assistant response found")
        
        decision = self._extract_json(agent_reply.content)
        
        return decision
    
    def update_experiment_memory(
        self,
        experiment: Dict[str, Any]
    ):
        """
        Update agent's memory with experiment results.
        
        Appends the experiment to the agent's experiment_log memory block
        so it can learn from past attempts.
        
        Args:
            experiment: Complete experiment record with proposal, results, decision
        
        Example:
            >>> strategist.update_experiment_memory({
            ...     "version": "v0.6",
            ...     "override": {"w_novelty": 0.35},
            ...     "results": {"delta_overall": 0.015},
            ...     "decision": "ACCEPT"
            ... })
        """
        # For now, just log the experiment
        # TODO: Fix Letta memory block API (blocks.create may have different signature)
        
        exp_summary = f"Experiment {experiment.get('version', 'N/A')}: "
        exp_summary += f"{experiment.get('decision', 'N/A')} - "
        exp_summary += f"Override: {experiment.get('override', {})}"
        
        print(f"📝 Logged experiment: {exp_summary}")
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        Get status of agent's memory blocks.
        
        Returns:
            Dict with memory block labels and sizes
        """
        blocks = list(self.client.agents.blocks.list(self.agent_id))
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "memory_blocks": [
                {
                    "label": block.label,
                    "size": len(block.value),
                    "preview": block.value[:100] + "..." if len(block.value) > 100 else block.value
                }
                for block in blocks
            ]
        }
    
    def _format_history(self, history: List[Dict]) -> str:
        """Format experiment history for agent message."""
        if not history:
            return "No previous experiments"
        
        # Show last 5 experiments
        recent = history[-5:]
        lines = []
        for exp in recent:
            lines.append(f"- {exp.get('version', 'N/A')}: {exp.get('decision', 'N/A')}")
            lines.append(f"  Override: {exp.get('override', {})}")
            lines.append(f"  Delta: {exp.get('delta_overall', 'N/A')}")
        
        return "\n".join(lines)
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from agent response text.
        
        Handles cases where JSON is embedded in markdown code blocks.
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from code block
        import re
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        
        # Try finding any JSON object
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        
        raise ValueError(f"Could not extract JSON from response: {text}")
