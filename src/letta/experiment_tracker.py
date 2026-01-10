"""
Experiment Tracker - External audit trail for Letta agent decisions.

Maintains a JSONL file with all experiment proposals, results, and decisions
for reproducibility and analysis.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ExperimentTracker:
    """
    Maintains external audit trail of policy evolution experiments.
    
    Logs all experiment proposals, results, and decisions to a JSONL file
    for:
    - Reproducibility
    - Debugging
    - Analysis of evolution patterns
    - Recovery from agent memory resets
    
    Example:
        >>> tracker = ExperimentTracker("outputs/agent_experiment_log.jsonl")
        >>> tracker.log_experiment({
        ...     "version": "v0.6",
        ...     "proposal": {...},
        ...     "results": {...},
        ...     "decision": "ACCEPT"
        ... })
    """
    
    def __init__(self, log_path: str = "outputs/agent_experiment_log.jsonl"):
        """
        Initialize experiment tracker.
        
        Args:
            log_path: Path to JSONL log file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file if doesn't exist
        if not self.log_path.exists():
            self.log_path.touch()
    
    def log_experiment(
        self,
        version: str,
        proposal: Dict[str, Any],
        results: Optional[Dict[str, Any]] = None,
        decision: Optional[str] = None,
        notes: str = ""
    ):
        """
        Log experiment to audit trail.
        
        Args:
            version: Policy version (e.g., "v0.6")
            proposal: Experiment proposal dict
            results: Experimental results dict
            decision: "ACCEPT" or "REJECT" or None if not decided yet
            notes: Additional notes
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "version": version,
            "proposal": proposal,
            "results": results,
            "decision": decision,
            "notes": notes
        }
        
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    
    def log_proposal(self, proposal: Dict[str, Any]):
        """Log just the proposal (before execution)."""
        self.log_experiment(
            version=proposal.get("version", "unknown"),
            proposal=proposal,
            results=None,
            decision=None,
            notes="Proposal generated"
        )
    
    def log_results(
        self,
        version: str,
        proposal: Dict[str, Any],
        results: Dict[str, Any]
    ):
        """Log experimental results (after execution, before decision)."""
        self.log_experiment(
            version=version,
            proposal=proposal,
            results=results,
            decision=None,
            notes="Results recorded"
        )
    
    def log_decision(
        self,
        version: str,
        proposal: Dict[str, Any],
        results: Dict[str, Any],
        decision: str,
        notes: str = ""
    ):
        """Log final decision (complete experiment)."""
        self.log_experiment(
            version=version,
            proposal=proposal,
            results=results,
            decision=decision,
            notes=notes or f"Decision: {decision}"
        )
    
    def load_history(self) -> list[Dict[str, Any]]:
        """
        Load all experiments from log file.
        
        Returns:
            List of experiment dicts
        """
        if not self.log_path.exists():
            return []
        
        experiments = []
        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    experiments.append(json.loads(line))
        
        return experiments
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of evolution history.
        
        Returns:
            Summary dict with counts and success rate
        """
        history = self.load_history()
        
        if not history:
            return {"total": 0, "accepted": 0, "rejected": 0, "success_rate": 0.0}
        
        accepted = sum(1 for exp in history if exp.get("decision") == "ACCEPT")
        rejected = sum(1 for exp in history if exp.get("decision") == "REJECT")
        pending = sum(1 for exp in history if exp.get("decision") is None)
        
        return {
            "total": len(history),
            "accepted": accepted,
            "rejected": rejected,
            "pending": pending,
            "success_rate": accepted / (accepted + rejected) if (accepted + rejected) > 0 else 0.0,
            "latest_version": history[-1].get("version") if history else None,
            "latest_decision": history[-1].get("decision") if history else None
        }
