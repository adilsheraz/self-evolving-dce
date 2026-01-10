"""
Self-Evolving Loop: Autonomous Policy Evolution Orchestrator

This module implements the autonomous self-evolving loop that combines:
1. Letta Policy Strategist (strategic planning)
2. DCE Engine (deterministic execution)
3. Evaluation pipeline (measure results)
4. Experiment tracking (audit trail)

Architecture:
    User Query → Retrieve → Compose (with current policy) → Evaluate
                              ↓
                    Policy Strategist (Letta) proposes new policy
                              ↓
                    Execute experiment with new policy
                              ↓
                    Agent decides: ACCEPT or REJECT
                              ↓
                    Update policy if ACCEPT, or try again if REJECT
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import copy

from ..letta import create_policy_strategist
from ..context import ContextComposer
from ..embedding import HuggingFaceEmbedder
from ..vectordb import PineconeLTM
from ..evaluation import ContextEvaluator
from ..config import Config


class SelfEvolvingLoop:
    """
    Orchestrates autonomous policy evolution using Letta agent.
    
    Workflow:
    1. Baseline: Evaluate current policy
    2. Propose: Letta suggests next experiment
    3. Execute: Run DCE with new policy
    4. Decide: Letta accepts or rejects
    5. Evolve: Update policy if accepted
    6. Repeat: Continue until convergence
    """
    
    def __init__(
        self,
        base_policy: Dict[str, float],
        letta_agent_id: str,
        pinecone_index_name: str,
        eval_queries: Optional[List[str]] = None,
        state_dir: Path = None
    ):
        """
        Initialize self-evolving loop.
        
        Args:
            base_policy: Initial DCE policy (w_novelty, w_redundancy, etc.)
            letta_agent_id: ID of your Letta policy strategist agent
            pinecone_index_name: Pinecone index for retrieval
            eval_queries: Test queries for evaluation (defaults to MDN queries)
            state_dir: Directory to save evolution state
        """
        self.base_policy = base_policy.copy()
        self.current_policy = base_policy.copy()
        
        # Initialize Letta strategist
        self.letta = create_policy_strategist(agent_id=letta_agent_id)
        
        # Initialize DCE components
        self.embedder = HuggingFaceEmbedder(
            model=Config.EMBEDDING_MODEL_NAME,
            api_key=Config.HF_TOKEN
        )
        
        self.vectordb = PineconeLTM(
            api_key=Config.PINECONE_API_KEY,
            index_name=pinecone_index_name
        )
        
        self.composer = ContextComposer()
        self.evaluator = ContextEvaluator()
        
        # Evaluation queries
        if eval_queries is None:
            self.eval_queries = [
                "How do you create HTML elements?",
                "What is the DOM?",
                "Explain CSS selectors",
                "How does JavaScript handle events?",
                "What are closures in JavaScript?"
            ]
        else:
            self.eval_queries = eval_queries
        
        # State tracking
        self.state_dir = state_dir or Path("state/evolution")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_history: List[Dict] = []
        self.current_version = "v0.1"
        
    def evaluate_policy(
        self,
        policy: Dict[str, float],
        top_k: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate a policy on test queries.
        
        Args:
            policy: DCE policy to evaluate
            top_k: Number of candidates to retrieve
            
        Returns:
            Dict with means and per-query scores
        """
        results = []
        
        for query in self.eval_queries:
            # Retrieve candidates
            query_vec = self.embedder.embed_query(query)
            candidates = self.vectordb.query(
                query_vector=query_vec,
                top_k=top_k,
                namespace="default"
            )
            
            # Extract indices
            candidate_idxs = [c["id"] for c in candidates]
            
            # Compose context with this policy
            context_pkg = self.composer.compose_context(
                query=query,
                candidate_idxs=candidate_idxs,
                query_vec=query_vec,
                policy=policy
            )
            
            # Evaluate
            scores = self.evaluator.evaluate_context(context_pkg)
            results.append({"query": query, **scores})
        
        df = pd.DataFrame(results)
        means = df[["relevance", "redundancy", "coverage", "efficiency", "overall"]].mean().to_dict()
        
        return {
            "means": means,
            "results": results,
            "n_queries": len(results)
        }
    
    def run_baseline(self) -> Dict[str, float]:
        """
        Evaluate current policy to establish baseline.
        
        Returns:
            Baseline metrics
        """
        print("\n" + "="*60)
        print("  BASELINE EVALUATION")
        print("="*60)
        
        eval_results = self.evaluate_policy(self.current_policy)
        baseline_metrics = eval_results["means"]
        
        print(f"\n📊 Baseline Metrics ({self.current_version}):")
        for metric, value in baseline_metrics.items():
            print(f"   {metric:12s}: {value:.3f}")
        
        # Save baseline
        baseline_path = self.state_dir / "baseline_metrics.json"
        with open(baseline_path, "w") as f:
            json.dump({
                "version": self.current_version,
                "policy": self.current_policy,
                "metrics": baseline_metrics
            }, f, indent=2)
        
        print(f"\n✅ Baseline saved to {baseline_path}")
        
        return baseline_metrics
    
    def propose_experiment(
        self,
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Ask Letta agent to propose next experiment.
        
        Args:
            baseline_metrics: Current baseline metrics
            
        Returns:
            Experiment proposal from Letta
        """
        print("\n" + "="*60)
        print("  PROPOSAL PHASE")
        print("="*60)
        
        print("\n🤖 Asking Letta for next experiment...")
        
        proposal = self.letta.propose_experiment(
            baseline_metrics=baseline_metrics,
            experiment_history=self.experiment_history
        )
        
        print(f"\n✅ Proposal received:")
        print(f"   Version: {proposal.get('version', 'N/A')}")
        print(f"   Override: {proposal.get('override', {})}")
        print(f"   Rationale: {proposal.get('rationale', 'N/A')[:100]}...")
        
        return proposal
    
    def execute_experiment(
        self,
        proposal: Dict[str, Any],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Execute the proposed experiment.
        
        Args:
            proposal: Experiment proposal from Letta
            baseline_metrics: Baseline to compute deltas
            
        Returns:
            Experiment results with deltas
        """
        print("\n" + "="*60)
        print("  EXECUTION PHASE")
        print("="*60)
        
        # Build candidate policy
        candidate_policy = self.current_policy.copy()
        candidate_policy.update(proposal["override"])
        candidate_policy["policy_version"] = proposal["version"]
        
        print(f"\n🧪 Testing policy {proposal['version']}...")
        print(f"   Override: {proposal['override']}")
        
        # Evaluate candidate
        eval_results = self.evaluate_policy(candidate_policy)
        candidate_metrics = eval_results["means"]
        
        # Compute deltas
        deltas = {
            f"delta_{k}": float(candidate_metrics[k] - baseline_metrics[k])
            for k in candidate_metrics
        }
        
        print(f"\n📈 Results:")
        for metric, delta in deltas.items():
            sign = "+" if delta >= 0 else ""
            print(f"   {metric:18s}: {sign}{delta:.4f}")
        
        return {
            "version": proposal["version"],
            "policy": candidate_policy,
            "metrics": candidate_metrics,
            "deltas": deltas,
            "proposal": proposal
        }
    
    def decide_on_experiment(
        self,
        proposal: Dict[str, Any],
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ask Letta agent to decide on experiment results.
        
        Args:
            proposal: Original experiment proposal
            results: Experiment results
            
        Returns:
            Decision from Letta (ACCEPT or REJECT)
        """
        print("\n" + "="*60)
        print("  DECISION PHASE")
        print("="*60)
        
        print("\n🎯 Asking Letta to decide...")
        
        decision = self.letta.decide_on_results(proposal, results)
        
        print(f"\n✅ Decision: {decision.get('decision', 'N/A')}")
        print(f"   Notes: {decision.get('notes', 'N/A')[:150]}...")
        
        return decision
    
    def update_policy(
        self,
        experiment: Dict[str, Any],
        decision: Dict[str, Any]
    ):
        """
        Update current policy if experiment was accepted.
        
        Args:
            experiment: Experiment results
            decision: Letta's decision
        """
        if decision["decision"] == "ACCEPT":
            print("\n✅ ACCEPTED - Updating policy...")
            
            self.current_policy = experiment["policy"].copy()
            self.current_version = experiment["version"]
            
            # Update Letta memory
            self.letta.update_experiment_memory(
                version=experiment["version"],
                override=experiment["proposal"]["override"],
                results=experiment["deltas"],
                decision="ACCEPT",
                notes=decision.get("notes", "")
            )
            
            # Save updated policy
            policy_path = self.state_dir / f"policy_{self.current_version}.json"
            with open(policy_path, "w") as f:
                json.dump({
                    "version": self.current_version,
                    "policy": self.current_policy,
                    "metrics": experiment["metrics"]
                }, f, indent=2)
            
            print(f"   Policy saved to {policy_path}")
            
        else:
            print("\n❌ REJECTED - Policy unchanged")
            
            # Still update Letta memory
            self.letta.update_experiment_memory(
                version=experiment["version"],
                override=experiment["proposal"]["override"],
                results=experiment["deltas"],
                decision="REJECT",
                notes=decision.get("notes", "")
            )
    
    def run_evolution_cycle(
        self,
        max_iterations: int = 10,
        convergence_threshold: float = 0.005
    ) -> Dict[str, Any]:
        """
        Run full evolution cycle with autonomous decision-making.
        
        Args:
            max_iterations: Max number of experiments
            convergence_threshold: Min improvement to continue
            
        Returns:
            Evolution summary
        """
        print("\n" + "="*70)
        print("  SELF-EVOLVING DCE: AUTONOMOUS POLICY EVOLUTION")
        print("="*70)
        
        # Step 1: Baseline
        baseline_metrics = self.run_baseline()
        
        accepted_count = 0
        rejected_count = 0
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n{'='*70}")
            print(f"  ITERATION {iteration}/{max_iterations}")
            print(f"{'='*70}")
            
            # Step 2: Propose
            proposal = self.propose_experiment(baseline_metrics)
            
            # Step 3: Execute
            experiment = self.execute_experiment(proposal, baseline_metrics)
            
            # Step 4: Decide
            decision = self.decide_on_experiment(proposal, experiment)
            
            # Step 5: Update
            self.update_policy(experiment, decision)
            
            # Track history
            self.experiment_history.append({
                "iteration": iteration,
                "version": experiment["version"],
                "override": proposal["override"],
                "deltas": experiment["deltas"],
                "decision": decision["decision"],
                "notes": decision.get("notes", "")
            })
            
            if decision["decision"] == "ACCEPT":
                accepted_count += 1
                baseline_metrics = experiment["metrics"]  # Update baseline
                
                # Check convergence
                if experiment["deltas"]["delta_overall"] < convergence_threshold:
                    print(f"\n🎉 Converged! Improvement < {convergence_threshold}")
                    break
            else:
                rejected_count += 1
        
        # Summary
        print("\n" + "="*70)
        print("  EVOLUTION SUMMARY")
        print("="*70)
        print(f"\nTotal Iterations: {iteration}")
        print(f"Accepted: {accepted_count}")
        print(f"Rejected: {rejected_count}")
        print(f"Final Version: {self.current_version}")
        print(f"Final Policy: {self.current_policy}")
        
        # Save history
        history_path = self.state_dir / "evolution_history.json"
        with open(history_path, "w") as f:
            json.dump(self.experiment_history, f, indent=2)
        
        print(f"\n✅ Evolution history saved to {history_path}")
        
        return {
            "iterations": iteration,
            "accepted": accepted_count,
            "rejected": rejected_count,
            "final_version": self.current_version,
            "final_policy": self.current_policy,
            "history": self.experiment_history
        }
