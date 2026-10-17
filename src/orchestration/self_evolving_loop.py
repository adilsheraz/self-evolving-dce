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
import re
import os
import tempfile
import numpy as np

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
        state_dir: Path = None,
        *, embedder=None, vectordb=None, strategist=None
    ):
        """
        Initialize self-evolving loop.

        Args:
            base_policy: Initial DCE composition policy
            letta_agent_id: ID of your Letta policy strategist agent
            pinecone_index_name: Pinecone index for retrieval
            eval_queries: Test queries for evaluation (defaults to MDN queries)
            state_dir: Directory to save evolution state
            embedder: Optional embedding provider (also used for offline tests)
            vectordb: Optional retrieval provider
            strategist: Optional policy strategist
        """
        self.base_policy = base_policy.copy()
        self.current_policy = base_policy.copy()

        # Initialize Letta strategist
        self.letta = strategist if strategist is not None else create_policy_strategist(agent_id=letta_agent_id)

        # Initialize DCE components
        self.embedder = embedder if embedder is not None else HuggingFaceEmbedder(
            model=Config.DEFAULT_EMBEDDING_MODEL,
            api_key=Config.HF_TOKEN
        )

        self.vectordb = vectordb if vectordb is not None else PineconeLTM(
            api_key=Config.PINECONE_API_KEY,
            index_name=pinecone_index_name
        )

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
            self.eval_queries = list(eval_queries)

        self.eval_queries = tuple(self.eval_queries)
        if not self.eval_queries or not all(isinstance(q, str) and q.strip() for q in self.eval_queries):
            raise ValueError("Evaluation queries must be nonempty strings")
        self._evaluation_inputs = None
        self._retrieval_top_k = None

        # State tracking
        self.state_dir = Path(state_dir or "state/evolution")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_history: List[Dict] = []
        self.current_version = self.current_policy.get("policy_version", "v0.1")

    def _save_json(self, name, payload):
        """Replace a local state file atomically; preserve the previous file on failure."""
        fd, temporary = tempfile.mkstemp(dir=self.state_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.state_dir / name)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def matches_to_chunks(matches, dimension):
        """Convert stored Pinecone metadata and values without re-embedding text."""
        chunks, vectors, seen = [], [], set()
        for match in matches:
            cid = match.get("id")
            metadata = dict(match.get("metadata") or {})
            if not isinstance(cid, str) or not cid or cid in seen:
                raise ValueError("Missing or duplicate Pinecone chunk ID")
            for field in ("doc_id", "title", "text"):
                if not isinstance(metadata.get(field), str) or not metadata[field].strip():
                    raise ValueError(f"Chunk {cid}: missing {field}")
            tokens = metadata.get("tokens")
            if isinstance(tokens, bool) or not isinstance(tokens, (int, float)) or not np.isfinite(tokens) or tokens <= 0 or int(tokens) != tokens:
                raise ValueError(f"Chunk {cid}: invalid token count")
            vector = np.asarray(match.get("values", []), dtype=float)
            if vector.shape != (dimension,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                raise ValueError(f"Chunk {cid}: invalid or missing stored embedding")
            metadata.update(chunk_id=cid, tokens=int(tokens))
            chunks.append(metadata)
            vectors.append(vector)
            seen.add(cid)
        return chunks, np.asarray(vectors, dtype=float).reshape(len(chunks), dimension)

    def _validate_proposal(self, proposal):
        version = proposal.get("version")
        override = proposal.get("override")
        if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", version):
            raise ValueError("Invalid policy version")
        if version == self.current_version or any(e["version"] == version for e in self.experiment_history):
            raise ValueError("Policy version already used")
        if not isinstance(override, dict) or not override or set(override) - (ContextComposer.DEFAULT_POLICY.keys() - {"policy_version"}):
            raise ValueError("Invalid policy override")
        effective = {**ContextComposer.DEFAULT_POLICY, **self.current_policy, **override}
        for key in ("budget_tokens", "min_chunk_tokens", "max_chunks", "max_chunks_per_doc"):
            value = effective[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Invalid policy parameter: {key}")
        for key in ("w_retrieval", "w_novelty", "w_doc_diversity", "redundancy_threshold", "redundancy_penalty"):
            value = effective[key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
                raise ValueError(f"Invalid policy parameter: {key}")
        # These existing parameters are currently unused by the composer.
        if set(override) & {"order_mode", "redundancy_penalty"}:
            raise ValueError("Cannot experiment with inactive composer parameters")

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

        if self._evaluation_inputs is None:
            inputs = []
            for query in self.eval_queries:
                query_vec = np.asarray(self.embedder.embed_query(query), dtype=float)
                if query_vec.ndim != 1 or not np.isfinite(query_vec).all() or np.linalg.norm(query_vec) == 0:
                    raise ValueError("Invalid query embedding")
                query_vec = query_vec / np.linalg.norm(query_vec)
                matches = self.vectordb.query(
                    query_vector=query_vec, top_k=top_k,
                    namespace=PineconeLTM.NS_CHUNKS,
                    include_metadata=True, include_values=True)
                chunks, embeddings = self.matches_to_chunks(matches, len(query_vec))
                inputs.append((query, query_vec, chunks, embeddings))
            # Publish only a complete snapshot; failed retrieval can be retried.
            self._save_json("evaluation_inputs.json", [
                {"query": q, "query_vector": v.tolist(), "chunks": c,
                 "embeddings": e.tolist(), "top_k": top_k,
                 "namespace": PineconeLTM.NS_CHUNKS,
                 "embedding_model": getattr(self.embedder, "model", None)}
                for q, v, c, e in inputs])
            self._evaluation_inputs = inputs
            self._retrieval_top_k = top_k
        if top_k != self._retrieval_top_k:
            raise ValueError("Cannot change top_k within a fixed evaluation snapshot")
        for query, query_vec, chunks, embeddings in self._evaluation_inputs:
            composer = ContextComposer(copy.deepcopy(chunks), embeddings.copy(), policy)
            context_pkg = composer.compose(query, list(range(len(chunks))), query_vec.copy())
            scores = self.evaluator.evaluate(context_pkg)
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

        print(f"\n Baseline Metrics ({self.current_version}):")
        for metric, value in baseline_metrics.items():
            print(f"   {metric:12s}: {value:.3f}")

        # Save baseline
        baseline_path = self.state_dir / "baseline_metrics.json"
        self._save_json("baseline_metrics.json", {
            "version": self.current_version, "policy": self.current_policy,
            "metrics": baseline_metrics})

        print(f"\n Baseline saved to {baseline_path}")

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

        print("\n Asking Letta for next experiment...")

        proposal = self.letta.propose_experiment(
            baseline_metrics=baseline_metrics,
            experiment_history=self.experiment_history,
            base_policy=self.current_policy
        )

        print(f"\n Proposal received:")
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

        self._validate_proposal(proposal)
        # Build candidate policy
        candidate_policy = self.current_policy.copy()
        candidate_policy.update(proposal["override"])
        candidate_policy["policy_version"] = proposal["version"]

        print(f"\n Testing policy {proposal['version']}...")
        print(f"   Override: {proposal['override']}")

        # Evaluate candidate
        eval_results = self.evaluate_policy(candidate_policy)
        candidate_metrics = eval_results["means"]

        # Compute deltas
        deltas = {
            f"delta_{k}": float(candidate_metrics[k] - baseline_metrics[k])
            for k in candidate_metrics
        }

        print(f"\n Results:")
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

        print("\n Asking Letta to decide...")

        decision = self.letta.decide_on_results(proposal, results)

        print(f"\n Decision: {decision.get('decision', 'N/A')}")
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
        outcome = decision.get("decision")
        if outcome not in ("ACCEPT", "REJECT"):
            raise ValueError("Decision must be ACCEPT or REJECT")
        record = {
            "version": experiment["version"],
            "override": experiment["proposal"]["override"],
            "results": experiment["deltas"], "decision": outcome,
            "notes": decision.get("notes", ""), "experiment": experiment}
        history = self.experiment_history + [record]
        policy = experiment["policy"] if outcome == "ACCEPT" else self.current_policy
        version = experiment["version"] if outcome == "ACCEPT" else self.current_version
        # Local checkpoint is authoritative. A remote notification failure cannot erase it.
        self._save_json("checkpoint.json", {
            "version": version, "policy": policy, "history": history})
        self.current_policy = copy.deepcopy(policy)
        self.current_version = version
        self.experiment_history = history
        self.letta.update_experiment_memory(record)

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

        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
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

            if decision["decision"] == "ACCEPT":
                accepted_count += 1
                baseline_metrics = experiment["metrics"]  # Update baseline

                # Check convergence
                if experiment["deltas"]["delta_overall"] < convergence_threshold:
                    print(f"\n Converged! Improvement < {convergence_threshold}")
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
        self._save_json("evolution_history.json", self.experiment_history)

        print(f"\n Evolution history saved to {history_path}")

        return {
            "iterations": iteration,
            "accepted": accepted_count,
            "rejected": rejected_count,
            "final_version": self.current_version,
            "final_policy": self.current_policy,
            "history": self.experiment_history
        }
