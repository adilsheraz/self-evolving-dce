"""
End-to-End Test: Full Policy Evolution Experiment

Tests the complete evolution workflow:
1. Define baseline and variant policies
2. Run same queries through all policies
3. Evaluate all contexts
4. Compare performance and identify improvements
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.embedding.embedder import HuggingFaceEmbedder
from src.context.composer import ContextComposer
from src.evaluation.evaluator import ContextEvaluator
from src.evolution.evolver import PolicyEvolver

print("="*60)
print("  End-to-End Policy Evolution Test")
print("="*60)

# Load existing chunks and embeddings
config = Config()
chunks_path = repo_root / "outputs" / "chunks" / "chunks.jsonl"
embeddings_path = repo_root / "outputs" / "embeddings" / "chunk_embeddings.npy"

print(f"\n📂 Loading data...")
chunks = []
with chunks_path.open("r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

embeddings = np.load(embeddings_path)
print(f"✅ Loaded {len(chunks)} chunks and {len(embeddings)} embeddings")

# Initialize embedder for query encoding
try:
    embedder = HuggingFaceEmbedder()
    print(f"✅ Embedder initialized: {embedder.model}")
except Exception as e:
    print(f"⚠️ Embedder initialization failed: {e}")
    print(f"   Using mock query vectors for testing")
    embedder = None

# ============================================================
# Step 1: Define Test Queries
# ============================================================
print(f"\n{'='*60}")
print("  Step 1: Define Evaluation Queries")
print(f"{'='*60}")

test_queries = [
    "What is semantic HTML?",
    "How does accessibility relate to HTML?",
    "Explain text-level semantics in HTML",
]

print(f"\n📋 Test queries ({len(test_queries)}):")
for i, q in enumerate(test_queries, 1):
    print(f"   {i}. {q}")

# ============================================================
# Step 2: Create Baseline Composer
# ============================================================
print(f"\n{'='*60}")
print("  Step 2: Initialize Baseline Composer")
print(f"{'='*60}")

baseline_composer = ContextComposer(chunks, embeddings)
baseline_policy = baseline_composer.policy

print(f"\n✅ Baseline Policy (v0.1):")
print(f"   Budget: {baseline_policy['budget_tokens']} tokens")
print(f"   Max chunks: {baseline_policy['max_chunks']}")
print(f"   Weights: retrieval={baseline_policy['w_retrieval']}, "
      f"novelty={baseline_policy['w_novelty']}, "
      f"diversity={baseline_policy['w_doc_diversity']}")

# ============================================================
# Step 3: Define Policy Variants
# ============================================================
print(f"\n{'='*60}")
print("  Step 3: Define Policy Variants")
print(f"{'='*60}")

evaluator = ContextEvaluator()
evolver = PolicyEvolver(baseline_policy, evaluator, improvement_threshold=0.01)

variants = {
    "v0.2_novelty_up": {
        "w_novelty": 0.30,
        "w_retrieval": 0.50,
        "w_doc_diversity": 0.20
    },
    "v0.3_strict_redundancy": {
        "redundancy_threshold": 0.85
    },
    "v0.4_larger_budget": {
        "budget_tokens": 1500
    }
}

print(f"\n🧬 Creating {len(variants)} policy variants:")
for name, changes in variants.items():
    print(f"   - {name}: {changes}")

evolved_policies = evolver.define_variants(variants)
print(f"\n✅ Created {len(evolved_policies)} variant policies")

# ============================================================
# Step 4: Run Evolution Experiment
# ============================================================
print(f"\n{'='*60}")
print("  Step 4: Run Full Evolution Experiment")
print(f"{'='*60}")

def compose_fn(query: str, policy: dict) -> dict:
    """Compose context for query using given policy."""
    # Create composer with this policy
    composer = ContextComposer(chunks, embeddings, policy=policy)
    
    # Get query vector
    if embedder:
        try:
            query_vec = embedder.embed_query(query)
        except Exception as e:
            print(f"\n⚠️ Embedding failed for '{query}': {e}")
            query_vec = embeddings[0]  # Fallback
    else:
        query_vec = embeddings[0]  # Mock
    
    # Get candidates (simulate retrieval with top chunks)
    candidate_idxs = list(range(50))
    
    # Compose context
    pkg = composer.compose(query, candidate_idxs, query_vec)
    return pkg

# Collect baseline results
print(f"\n📊 Running baseline (v0.1)...")
baseline_packages = []
for query in test_queries:
    pkg = compose_fn(query, baseline_policy)
    baseline_packages.append(pkg)
    print(f"   ✓ '{query[:40]}...' → {len(pkg['selected'])} chunks, "
          f"{pkg['used_tokens']} tokens")

baseline_results = evaluator.evaluate_batch(baseline_packages)
baseline_df = pd.DataFrame(baseline_results)

print(f"\n✅ Baseline evaluation complete")
print(f"   Mean overall score: {baseline_df['overall'].mean():.3f}")

# Run experiment with variants
print(f"\n🧬 Running experiment with {len(evolved_policies)} variants...")

experiment_results = evolver.run_experiment(
    queries=test_queries,
    policies=evolved_policies,
    compose_fn=compose_fn
)

print(f"\n✅ Experiment complete: {len(experiment_results)} results")

# ============================================================
# Step 5: Compare Policies
# ============================================================
print(f"\n{'='*60}")
print("  Step 5: Compare Against Baseline")
print(f"{'='*60}")

summary_df, delta_df = evolver.compare_policies(
    baseline_df=baseline_df,
    experiment_df=experiment_results,
    baseline_version="v0.1"
)

print(f"\n📊 Policy Performance Summary:")
print(summary_df.to_string(index=False))

print(f"\n📈 Delta from Baseline:")
print(delta_df.to_string(index=False))

# ============================================================
# Step 6: Select Improved Policies
# ============================================================
print(f"\n{'='*60}")
print("  Step 6: Select Improved Policies")
print(f"{'='*60}")

improved_df = evolver.select_improved_policies(delta_df)

if not improved_df.empty:
    print(f"\n✅ Found {len(improved_df)} improved policies:")
    print(improved_df[['policy_version', 'delta_overall', 'delta_relevance', 
                       'delta_coverage']].to_string(index=False))
else:
    print(f"\n⚠️ No policies exceeded improvement threshold")
    print(f"   Baseline v0.1 is stable for this dataset")

# ============================================================
# Step 7: Save Results
# ============================================================
print(f"\n{'='*60}")
print("  Step 7: Save Evolution Artifacts")
print(f"{'='*60}")

output_dir = repo_root / "outputs" / "evaluations"
evolver.save_results(experiment_results, delta_df, output_dir, prefix="test_evolution")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("  ✅ Full Evolution Test Complete!")
print(f"{'='*60}")

print(f"\n📊 Test Summary:")
print(f"   Queries tested: {len(test_queries)}")
print(f"   Policies evaluated: {len(evolved_policies) + 1} (baseline + {len(evolved_policies)} variants)")
print(f"   Total contexts composed: {len(baseline_packages) + len(experiment_results)}")
print(f"   Total evaluations: {len(baseline_results) + len(experiment_results)}")
print(f"   Improved policies found: {len(improved_df)}")

print(f"\n💡 Key Insights:")
print(f"   Best performing policy: {summary_df.loc[summary_df['overall'].idxmax(), 'policy_version']}")
print(f"   Best overall score: {summary_df['overall'].max():.3f}")
print(f"   Baseline score: {baseline_df['overall'].mean():.3f}")

print(f"\n🎉 End-to-end policy evolution workflow verified!")
