"""
Context Quality Evaluation

Evaluates context packages along four axes:
1. Relevance - semantic alignment with query
2. Redundancy - semantic overlap between chunks (lower is better)
3. Coverage - topic breadth and diversity
4. Efficiency - token budget utilization

No answer generation, no LLM judging - purely objective metrics.
"""

import numpy as np
from typing import Dict, Any, List


# Default evaluation weights (frozen for v0.1)
EVAL_WEIGHTS = {
    "relevance": 0.35,
    "redundancy": 0.25,
    "coverage": 0.20,
    "efficiency": 0.20,
}


class ContextEvaluator:
    """
    Evaluate context package quality using objective metrics.
    
    Metrics:
    - Relevance: Average retrieval similarity to query
    - Redundancy: Semantic overlap between selected chunks (inverted)
    - Coverage: Topic breadth via document diversity and chunk count
    - Efficiency: Token budget utilization
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize evaluator.
        
        Args:
            weights: Metric weights for overall score (uses defaults if not provided)
        """
        self.weights = {**EVAL_WEIGHTS, **(weights or {})}
    
    @staticmethod
    def eval_relevance(pkg: Dict[str, Any]) -> float:
        """
        Evaluate relevance: average retrieval similarity across selected chunks.
        
        Args:
            pkg: Context package from ContextComposer
            
        Returns:
            Relevance score (0-1)
        """
        selected = pkg["selected"]
        if not selected:
            return 0.0
        return float(np.mean([s["score_parts"]["retrieval"] for s in selected]))
    
    @staticmethod
    def eval_redundancy(pkg: Dict[str, Any]) -> float:
        """
        Evaluate redundancy: semantic overlap between chunks.
        
        Returns non-redundancy score (higher is better).
        Based on worst-case cosine similarity between any two chunks.
        
        Args:
            pkg: Context package from ContextComposer
            
        Returns:
            Non-redundancy score (0-1, higher means less redundant)
        """
        worst = pkg["composer_metrics"]["worst_redundancy"]
        return float(1.0 - worst)
    
    @staticmethod
    def eval_coverage(pkg: Dict[str, Any]) -> float:
        """
        Evaluate coverage: topic breadth and diversity.
        
        Combines:
        - Document diversity ratio (unique docs / total chunks)
        - Chunk count saturation (rewards more chunks with diminishing returns)
        
        Args:
            pkg: Context package from ContextComposer
            
        Returns:
            Coverage score (0-1)
        """
        selected = pkg["selected"]
        if not selected:
            return 0.0
        
        unique_docs = len(set(s["doc_id"] for s in selected))
        total = len(selected)
        
        # Document diversity ratio
        doc_div = unique_docs / total
        
        # Mild reward for more chunks (diminishing returns)
        chunk_factor = min(1.0, total / 6)
        
        return float(0.6 * doc_div + 0.4 * chunk_factor)
    
    @staticmethod
    def eval_efficiency(pkg: Dict[str, Any]) -> float:
        """
        Evaluate efficiency: token budget utilization.
        
        Penalizes severe underuse (< 50% budget).
        
        Args:
            pkg: Context package from ContextComposer
            
        Returns:
            Efficiency score (0-1)
        """
        used = pkg["used_tokens"]
        budget = pkg["budget_tokens"]
        
        if budget <= 0:
            return 0.0
        
        ratio = used / budget
        
        # Penalize heavy underuse
        if ratio < 0.5:
            return float(ratio * 0.5)
        
        return float(ratio)
    
    def evaluate(self, pkg: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate context package across all metrics.
        
        Args:
            pkg: Context package from ContextComposer
            
        Returns:
            Dictionary with individual and overall scores
        """
        scores = {
            "relevance": self.eval_relevance(pkg),
            "redundancy": self.eval_redundancy(pkg),
            "coverage": self.eval_coverage(pkg),
            "efficiency": self.eval_efficiency(pkg),
        }
        
        overall = sum(scores[k] * self.weights[k] for k in scores)
        scores["overall"] = float(overall)
        
        return scores
    
    def evaluate_batch(self, packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple context packages.
        
        Args:
            packages: List of context packages
            
        Returns:
            List of evaluation results with query and scores
        """
        results = []
        for pkg in packages:
            scores = self.evaluate(pkg)
            results.append({
                "query": pkg["query"],
                "policy_version": pkg.get("policy_version", "unknown"),
                **scores,
                "selected_chunks": len(pkg.get("selected", [])),
                "used_tokens": pkg.get("used_tokens", 0),
            })
        return results
