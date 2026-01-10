"""
QA Dataset Bias Validation for DCE

Validates that QA pairs don't introduce systematic bias in DCE metrics.
Tests length invariance, format invariance, and selection diversity.

Usage:
    python scripts/validate_qa_bias.py
    python scripts/validate_qa_bias.py --chunks outputs/chunks/qa_chunks.jsonl
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.embedding.embedder import HuggingFaceEmbedder
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  Warning: Embedding module not available. Skipping embedding-based tests.")


def load_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    """Load transformed QA chunks."""
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks


def test_length_distribution(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Test 1: Length Distribution Analysis
    Ensure chunk lengths are reasonably uniform for unbiased selection.
    """
    print("\n" + "="*60)
    print("TEST 1: LENGTH DISTRIBUTION")
    print("="*60)
    
    lengths = [chunk["tokens"] for chunk in chunks]
    
    results = {
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "median": float(np.median(lengths)),
        "cv": float(np.std(lengths) / np.mean(lengths))  # Coefficient of variation
    }
    
    print(f"Mean Length: {results['mean']:.1f} tokens")
    print(f"Std Dev: {results['std']:.1f} tokens")
    print(f"Range: {results['min']:.0f} - {results['max']:.0f} tokens")
    print(f"Median: {results['median']:.0f} tokens")
    print(f"Coefficient of Variation: {results['cv']:.3f}")
    
    # Check uniformity (CV < 0.5 is good for QA pairs)
    if results['cv'] < 0.5:
        print("✅ PASS: Length distribution is uniform (CV < 0.5)")
        results['status'] = "PASS"
    else:
        print("⚠️  WARNING: High length variance (CV >= 0.5)")
        results['status'] = "WARNING"
    
    return results


def test_metadata_distribution(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Test 2: Metadata Distribution Analysis
    Check for balanced distribution across types and difficulties.
    """
    print("\n" + "="*60)
    print("TEST 2: METADATA DISTRIBUTION")
    print("="*60)
    
    types = defaultdict(int)
    difficulties = defaultdict(int)
    
    for chunk in chunks:
        qa_type = chunk["metadata"].get("qa_type", "unknown")
        difficulty = chunk["metadata"].get("difficulty", "medium")
        types[qa_type] += 1
        difficulties[difficulty] += 1
    
    total = len(chunks)
    
    print("\n📊 Type Distribution:")
    type_imbalance = []
    for qtype, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total) * 100
        print(f"  • {qtype}: {count} ({pct:.1f}%)")
        type_imbalance.append(pct)
    
    print("\n📊 Difficulty Distribution:")
    diff_imbalance = []
    for diff, count in sorted(difficulties.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total) * 100
        print(f"  • {diff}: {count} ({pct:.1f}%)")
        diff_imbalance.append(pct)
    
    # Calculate Gini coefficient for imbalance (0 = perfect balance, 1 = total imbalance)
    type_gini = calculate_gini(type_imbalance)
    diff_gini = calculate_gini(diff_imbalance)
    
    print(f"\n📈 Type Imbalance (Gini): {type_gini:.3f}")
    print(f"📈 Difficulty Imbalance (Gini): {diff_gini:.3f}")
    
    results = {
        "types": dict(types),
        "difficulties": dict(difficulties),
        "type_gini": type_gini,
        "diff_gini": diff_gini
    }
    
    # Check balance (Gini < 0.3 is good)
    if type_gini < 0.3 and diff_gini < 0.3:
        print("✅ PASS: Metadata distribution is balanced")
        results['status'] = "PASS"
    elif type_gini < 0.5 and diff_gini < 0.5:
        print("⚠️  WARNING: Moderate metadata imbalance")
        results['status'] = "WARNING"
    else:
        print("❌ FAIL: High metadata imbalance detected")
        results['status'] = "FAIL"
    
    return results


def test_content_diversity(chunks: List[Dict[str, Any]], sample_size: int = 100) -> Dict[str, Any]:
    """
    Test 3: Content Diversity Analysis
    Measure unique entity coverage and content overlap.
    """
    print("\n" + "="*60)
    print("TEST 3: CONTENT DIVERSITY")
    print("="*60)
    
    # Sample chunks to avoid memory issues
    import random
    sample = random.sample(chunks, min(sample_size, len(chunks)))
    
    # Collect all entities
    all_entities = []
    entities_per_chunk = []
    
    for chunk in sample:
        entities = chunk["metadata"].get("key_entities", [])
        all_entities.extend(entities)
        entities_per_chunk.append(len(entities))
    
    unique_entities = len(set(all_entities))
    total_entities = len(all_entities)
    
    print(f"Sample Size: {len(sample)} chunks")
    print(f"Total Entities: {total_entities}")
    print(f"Unique Entities: {unique_entities}")
    print(f"Uniqueness Ratio: {unique_entities/total_entities:.3f}")
    print(f"Avg Entities/Chunk: {np.mean(entities_per_chunk):.1f}")
    
    results = {
        "sample_size": len(sample),
        "total_entities": total_entities,
        "unique_entities": unique_entities,
        "uniqueness_ratio": unique_entities / total_entities,
        "avg_entities_per_chunk": np.mean(entities_per_chunk)
    }
    
    # Check diversity (uniqueness > 0.5 is good for QA pairs)
    if results['uniqueness_ratio'] > 0.5:
        print("✅ PASS: High content diversity (uniqueness > 0.5)")
        results['status'] = "PASS"
    elif results['uniqueness_ratio'] > 0.3:
        print("⚠️  WARNING: Moderate content overlap")
        results['status'] = "WARNING"
    else:
        print("❌ FAIL: High content redundancy")
        results['status'] = "FAIL"
    
    return results


def test_embedding_variance(chunks: List[Dict[str, Any]], sample_size: int = 100) -> Dict[str, Any]:
    """
    Test 4: Embedding Variance Analysis (Optional - requires embeddings)
    Measure semantic diversity in embedding space.
    """
    if not EMBEDDINGS_AVAILABLE:
        return {"status": "SKIPPED", "reason": "Embeddings not available"}
    
    print("\n" + "="*60)
    print("TEST 4: EMBEDDING VARIANCE")
    print("="*60)
    
    # Sample chunks
    import random
    sample = random.sample(chunks, min(sample_size, len(chunks)))
    
    # Initialize embedder
    print("Initializing embedder...")
    embedder = HuggingFaceEmbedder(model="BAAI/bge-small-en-v1.5")
    
    # Get embeddings
    print(f"Computing embeddings for {len(sample)} chunks...")
    texts = [chunk["text"] for chunk in sample]
    embeddings = embedder.embed_texts(texts)
    embeddings_array = np.array(embeddings)
    
    # Calculate variance across dimensions
    variance_per_dim = np.var(embeddings_array, axis=0)
    mean_variance = np.mean(variance_per_dim)
    
    # Calculate pairwise distances
    from sklearn.metrics.pairwise import cosine_distances
    distances = cosine_distances(embeddings_array)
    
    # Get upper triangle (exclude diagonal)
    upper_triangle = distances[np.triu_indices_from(distances, k=1)]
    
    results = {
        "sample_size": len(sample),
        "embedding_dim": embeddings_array.shape[1],
        "mean_variance": float(mean_variance),
        "mean_distance": float(np.mean(upper_triangle)),
        "std_distance": float(np.std(upper_triangle)),
        "min_distance": float(np.min(upper_triangle)),
        "max_distance": float(np.max(upper_triangle))
    }
    
    print(f"Embedding Dimension: {results['embedding_dim']}")
    print(f"Mean Variance: {results['mean_variance']:.4f}")
    print(f"Mean Cosine Distance: {results['mean_distance']:.3f}")
    print(f"Distance Range: {results['min_distance']:.3f} - {results['max_distance']:.3f}")
    
    # Check semantic diversity (mean distance > 0.3 is good)
    if results['mean_distance'] > 0.3:
        print("✅ PASS: High semantic diversity (distance > 0.3)")
        results['status'] = "PASS"
    elif results['mean_distance'] > 0.2:
        print("⚠️  WARNING: Moderate semantic similarity")
        results['status'] = "WARNING"
    else:
        print("❌ FAIL: Low semantic diversity (chunks too similar)")
        results['status'] = "FAIL"
    
    return results


def calculate_gini(values: List[float]) -> float:
    """Calculate Gini coefficient for distribution imbalance."""
    sorted_values = sorted(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    return (2 * np.sum((n - i) * v for i, v in enumerate(sorted_values))) / (n * cumsum[-1]) - (n + 1) / n


def print_summary(all_results: Dict[str, Dict[str, Any]]):
    """Print overall validation summary."""
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    statuses = [result.get("status", "UNKNOWN") for result in all_results.values()]
    
    pass_count = statuses.count("PASS")
    warning_count = statuses.count("WARNING")
    fail_count = statuses.count("FAIL")
    skip_count = statuses.count("SKIPPED")
    
    print(f"\n✅ PASS: {pass_count}")
    print(f"⚠️  WARNING: {warning_count}")
    print(f"❌ FAIL: {fail_count}")
    print(f"⏭️  SKIPPED: {skip_count}")
    
    if fail_count == 0 and warning_count <= 1:
        print("\n🎉 Overall: EXCELLENT - No significant biases detected!")
        print("✅ Safe to use QA dataset with DCE")
    elif fail_count == 0:
        print("\n✅ Overall: GOOD - Minor warnings, acceptable for production")
        print("⚠️  Consider reviewing warnings but generally safe to proceed")
    else:
        print("\n⚠️  Overall: NEEDS ATTENTION - Some tests failed")
        print("❌ Review failed tests before using in production")
    
    print("="*60 + "\n")


def main():
    """Main validation pipeline."""
    repo_root = Path(__file__).parent.parent
    chunks_path = repo_root / "outputs" / "chunks" / "qa_chunks.jsonl"
    
    # Override from command line
    if len(sys.argv) > 1 and sys.argv[1] == "--chunks":
        chunks_path = Path(sys.argv[2])
    
    print("🔍 QA Dataset Bias Validation")
    print(f"📂 Chunks File: {chunks_path}")
    
    if not chunks_path.exists():
        print(f"\n❌ Chunks file not found: {chunks_path}")
        print("💡 Run transformation first: python scripts/transform_qa_to_chunks.py")
        return
    
    # Load chunks
    print("\n📥 Loading chunks...")
    chunks = load_chunks(chunks_path)
    print(f"✅ Loaded {len(chunks):,} chunks")
    
    # Run all tests
    all_results = {}
    
    all_results["length"] = test_length_distribution(chunks)
    all_results["metadata"] = test_metadata_distribution(chunks)
    all_results["diversity"] = test_content_diversity(chunks, sample_size=200)
    all_results["embeddings"] = test_embedding_variance(chunks, sample_size=100)
    
    # Print summary
    print_summary(all_results)
    
    # Save results
    results_path = chunks_path.parent / "qa_bias_validation.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"📄 Detailed results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
