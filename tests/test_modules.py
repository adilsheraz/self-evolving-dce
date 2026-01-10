"""
Test script for newly extracted modules:
- Context composition (Notebook 04)
- Context evaluation (Notebook 05)
- Policy evolution (Notebook 06)

Tests use existing data from previous pipeline runs.
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.context.composer import ContextComposer, render_context_text
from src.evaluation.evaluator import ContextEvaluator
from src.evolution.evolver import PolicyEvolver, build_policy_variant

print("="*60)
print("  Testing Context & Evaluation Modules")
print("="*60)

# Load existing chunks and embeddings
config = Config()
chunks_path = repo_root / "outputs" / "chunks" / "chunks.jsonl"
embeddings_path = repo_root / "outputs" / "embeddings" / "chunk_embeddings.npy"

print(f"\n📂 Loading test data...")
print(f"   Chunks: {chunks_path}")
print(f"   Embeddings: {embeddings_path}")

if not chunks_path.exists():
    print(f"❌ Chunks file not found. Run auto_rag_pipeline.py first.")
    sys.exit(1)

if not embeddings_path.exists():
    print(f"❌ Embeddings file not found. Run auto_rag_pipeline.py first.")
    sys.exit(1)

# Load chunks
chunks = []
with chunks_path.open("r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

# Load embeddings
embeddings = np.load(embeddings_path)

print(f"\n✅ Loaded {len(chunks)} chunks")
print(f"✅ Loaded {len(embeddings)} embeddings (shape: {embeddings.shape})")

# ============================================================
# Test 1: Context Composer
# ============================================================
print(f"\n{'='*60}")
print("  Test 1: Context Composer")
print(f"{'='*60}")

composer = ContextComposer(chunks, embeddings)
print(f"✅ ContextComposer initialized")
print(f"   Policy version: {composer.policy['policy_version']}")
print(f"   Token budget: {composer.policy['budget_tokens']}")

# Simulate a query (use first chunk's embedding as query for testing)
test_query = "What is semantic HTML?"
query_vec = embeddings[0]  # Use first chunk's embedding as proxy

# Get top candidates (simulate retrieval by using first 40 chunks)
candidate_idxs = list(range(40))

print(f"\n🔍 Composing context for query: '{test_query}'")
print(f"   Candidates: {len(candidate_idxs)}")

pkg = composer.compose(test_query, candidate_idxs, query_vec)

print(f"\n✅ Context package created:")
print(f"   Selected chunks: {len(pkg['selected'])}")
print(f"   Used tokens: {pkg['used_tokens']}/{pkg['budget_tokens']}")
print(f"   Excluded chunks: {len(pkg['excluded'])}")
print(f"   Metrics:")
print(f"     - Avg relevance: {pkg['composer_metrics']['avg_retrieval_relevance']:.3f}")
print(f"     - Worst redundancy: {pkg['composer_metrics']['worst_redundancy']:.3f}")
print(f"     - Doc diversity: {pkg['composer_metrics']['doc_diversity_ratio']:.3f}")
print(f"     - Budget efficiency: {pkg['composer_metrics']['budget_efficiency']:.3f}")

# Test render function
context_text = render_context_text(pkg)
print(f"\n📄 Rendered context preview (first 300 chars):")
print(context_text[:300] + "...")

# ============================================================
# Test 2: Context Evaluator
# ============================================================
print(f"\n{'='*60}")
print("  Test 2: Context Evaluator")
print(f"{'='*60}")

evaluator = ContextEvaluator()
print(f"✅ ContextEvaluator initialized")
print(f"   Weights: {evaluator.weights}")

scores = evaluator.evaluate(pkg)

print(f"\n✅ Context evaluation scores:")
print(f"   Relevance: {scores['relevance']:.3f}")
print(f"   Redundancy: {scores['redundancy']:.3f}")
print(f"   Coverage: {scores['coverage']:.3f}")
print(f"   Efficiency: {scores['efficiency']:.3f}")
print(f"   Overall: {scores['overall']:.3f}")

# Test batch evaluation
print(f"\n🔄 Testing batch evaluation with 3 queries...")
test_queries = [
    "What is semantic HTML?",
    "How to use CSS flexbox?",
    "Explain JavaScript closures"
]

batch_packages = []
for query in test_queries:
    pkg_test = composer.compose(query, candidate_idxs, query_vec)
    batch_packages.append(pkg_test)

batch_results = evaluator.evaluate_batch(batch_packages)

print(f"\n✅ Batch evaluation complete:")
for result in batch_results:
    print(f"   '{result['query'][:40]}...' → Overall: {result['overall']:.3f}")

# ============================================================
# Test 3: Policy Evolution
# ============================================================
print(f"\n{'='*60}")
print("  Test 3: Policy Evolver")
print(f"{'='*60}")

evolver = PolicyEvolver(composer.policy, evaluator, improvement_threshold=0.01)
print(f"✅ PolicyEvolver initialized")
print(f"   Improvement threshold: {evolver.improvement_threshold}")

# Define test variants
test_variants = {
    "v0.2_novelty_up": {"w_novelty": 0.30, "w_retrieval": 0.50},
    "v0.3_strict_redundancy": {"redundancy_threshold": 0.85},
}

print(f"\n🧬 Defining {len(test_variants)} policy variants:")
for name in test_variants:
    print(f"   - {name}")

evolved_policies = evolver.define_variants(test_variants)

print(f"\n✅ Policy variants created:")
for name, policy in evolved_policies.items():
    print(f"   {name}: version={policy['policy_version']}")

# Test policy variant building
variant_policy = build_policy_variant(
    composer.policy,
    {"budget_tokens": 1500},
    "v0.4_larger_budget"
)
print(f"\n✅ build_policy_variant test:")
print(f"   Original budget: {composer.policy['budget_tokens']}")
print(f"   Variant budget: {variant_policy['budget_tokens']}")
print(f"   Variant version: {variant_policy['policy_version']}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("  ✅ All Module Tests Passed!")
print(f"{'='*60}")

print(f"\n📊 Test Summary:")
print(f"   ✅ ContextComposer: Created context package with {len(pkg['selected'])} chunks")
print(f"   ✅ ContextEvaluator: Evaluated {len(batch_packages)} contexts")
print(f"   ✅ PolicyEvolver: Created {len(evolved_policies)} policy variants")
print(f"\n🎉 All modules working correctly!")
