"""
Policy Evolution and Self-Improvement

Systematically evolve composition policies using controlled experiments:
- One-parameter-at-a-time changes
- Objective metric comparisons
- Conservative improvement thresholds
- Reversible decisions

No heuristic tuning, no answer generation, no LLM judging.
"""

import copy
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Callable, Tuple
from pathlib import Path


def build_policy_variant(base: Dict[str, Any], override: Dict[str, Any], version: str) -> Dict[str, Any]:
    """
    Create policy variant by safely merging overrides into base policy.
    
    Args:
        base: Base policy dictionary
        override: Parameters to override
        version: Version identifier for the variant
        
    Returns:
        New policy dictionary with overrides applied
    """
    policy = copy.deepcopy(base)
    for k, v in override.items():
        policy[k] = v
    policy["policy_version"] = version
    return policy


class PolicyEvolver:
    """
    Evolve context composition policies through controlled experiments.
    
    Strategy:
    1. Define policy variants (one change at a time)
    2. Run same queries through all policies
    3. Evaluate contexts with objective metrics
    4. Compare against baseline
    5. Select improved policies based on evidence
    """
    
    DEFAULT_IMPROVEMENT_THRESHOLD = 0.01  # Conservative threshold
    
    def __init__(
        self,
        base_policy: Dict[str, Any],
        evaluator: Any,
        improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD
    ):
        """
        Initialize policy evolver.
        
        Args:
            base_policy: Baseline policy to evolve from
            evaluator: ContextEvaluator instance
            improvement_threshold: Minimum improvement to accept variant
        """
        self.base_policy = base_policy
        self.evaluator = evaluator
        self.improvement_threshold = improvement_threshold
    
    def define_variants(self, variants: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Define policy variants from override specifications.
        
        Args:
            variants: Dict mapping version names to parameter overrides
            
        Returns:
            Dict mapping version names to complete policy dictionaries
        """
        return {
            name: build_policy_variant(self.base_policy, changes, name)
            for name, changes in variants.items()
        }
    
    def run_experiment(
        self,
        queries: List[str],
        policies: Dict[str, Dict[str, Any]],
        compose_fn: Callable[[str, Dict[str, Any]], Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Run controlled experiment: evaluate all policies on same queries.
        
        Args:
            queries: List of evaluation queries
            policies: Dict mapping version names to policies
            compose_fn: Function that takes (query, policy) and returns context package
            
        Returns:
            DataFrame with evaluation results for all policy-query pairs
        """
        results = []
        
        for version, policy in policies.items():
            print(f"🧬 Evaluating policy: {version}")
            
            for query in queries:
                # Compose context with this policy
                pkg = compose_fn(query, policy)
                
                # Evaluate context
                scores = self.evaluator.evaluate(pkg)
                
                results.append({
                    "query": query,
                    "policy_version": version,
                    **scores,
                    "selected_chunks": len(pkg.get("selected", [])),
                    "used_tokens": pkg.get("used_tokens", 0),
                })
        
        return pd.DataFrame(results)
    
    def compare_policies(
        self,
        baseline_df: pd.DataFrame,
        experiment_df: pd.DataFrame,
        baseline_version: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compare experimental policies against baseline.
        
        Args:
            baseline_df: Evaluation results for baseline policy
            experiment_df: Evaluation results for experimental policies
            baseline_version: Policy version identifier for baseline
            
        Returns:
            Tuple of (summary_df, delta_df)
            - summary_df: Mean scores per policy
            - delta_df: Delta from baseline per policy
        """
        # Ensure baseline has correct version
        baseline_df = baseline_df.copy()
        baseline_df["policy_version"] = baseline_version
        
        # Combine and summarize
        df_all = pd.concat([baseline_df, experiment_df], ignore_index=True)
        
        summary = df_all.groupby("policy_version")[[
            "relevance", "redundancy", "coverage", "efficiency", "overall"
        ]].mean().reset_index()
        
        # Compute deltas from baseline
        baseline_row = summary[summary["policy_version"] == baseline_version].iloc[0]
        
        delta_rows = []
        for _, row in summary.iterrows():
            if row["policy_version"] == baseline_version:
                continue
            
            delta = {
                "policy_version": row["policy_version"],
                "delta_relevance": row["relevance"] - baseline_row["relevance"],
                "delta_redundancy": row["redundancy"] - baseline_row["redundancy"],
                "delta_coverage": row["coverage"] - baseline_row["coverage"],
                "delta_efficiency": row["efficiency"] - baseline_row["efficiency"],
                "delta_overall": row["overall"] - baseline_row["overall"],
            }
            delta_rows.append(delta)
        
        delta_df = pd.DataFrame(delta_rows)
        delta_df = delta_df.sort_values("delta_overall", ascending=False)
        
        return summary, delta_df
    
    def select_improved_policies(
        self,
        delta_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Select policies that exceed improvement threshold.
        
        Args:
            delta_df: Delta dataframe from compare_policies
            
        Returns:
            DataFrame with only improved policies
        """
        improved = delta_df[delta_df["delta_overall"] > self.improvement_threshold]
        
        if improved.empty:
            print(f"⚠️ No policy exceeds improvement threshold ({self.improvement_threshold:.4f}) — baseline retained")
        else:
            print(f"✅ Found {len(improved)} improved policies:")
            print(improved[["policy_version", "delta_overall"]].to_string(index=False))
        
        return improved
    
    def save_results(
        self,
        experiment_df: pd.DataFrame,
        delta_df: pd.DataFrame,
        output_dir: Path,
        prefix: str = "policy_evolution"
    ):
        """
        Save evolution experiment artifacts.
        
        Args:
            experiment_df: Full experiment results
            delta_df: Delta analysis
            output_dir: Output directory
            prefix: Filename prefix
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / f"{prefix}_results.csv"
        delta_path = output_dir / f"{prefix}_deltas.csv"
        
        experiment_df.to_csv(results_path, index=False)
        delta_df.to_csv(delta_path, index=False)
        
        print(f"✅ Saved evolution artifacts:")
        print(f"   - {results_path}")
        print(f"   - {delta_path}")
