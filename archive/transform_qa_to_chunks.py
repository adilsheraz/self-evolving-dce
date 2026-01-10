"""
Transform QA Pairs Dataset to DCE-Compatible Chunks

Converts the new_dataset QA pairs into chunk format for DCE ingestion.
Handles all 11 PDFs with 5,662 total QA pairs.

Usage:
    python scripts/transform_qa_to_chunks.py
    python scripts/transform_qa_to_chunks.py --output outputs/chunks/qa_chunks.jsonl
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_qa_pairs(qa_dir: Path) -> List[Dict[str, Any]]:
    """Load all QA pairs from the new_dataset/qa_pairs directory."""
    all_qa_pairs = []
    
    for qa_file in sorted(qa_dir.glob("*.jsonl")):
        print(f"Loading {qa_file.name}...")
        with open(qa_file, 'r', encoding='utf-8') as f:
            for line in f:
                qa = json.loads(line.strip())
                all_qa_pairs.append(qa)
    
    return all_qa_pairs


def transform_qa_to_chunk(qa: Dict[str, Any], chunk_id: int) -> Dict[str, Any]:
    """
    Transform a QA pair into DCE chunk format.
    
    Args:
        qa: QA pair with keys: document, page, question, answer, type, difficulty, key_entities
        chunk_id: Unique chunk identifier
    
    Returns:
        DCE-compatible chunk with text and metadata
    """
    # Combine question and answer into chunk text
    text = f"Q: {qa['question']}\nA: {qa['answer']}"
    
    # Create metadata
    metadata = {
        "document": qa.get("document", "unknown"),
        "page": qa.get("page", 0),
        "qa_type": qa.get("type", "unknown"),
        "difficulty": qa.get("difficulty", "medium"),
        "key_entities": qa.get("key_entities", []),
        "generated_at": qa.get("generated_at", "")
    }
    
    # Calculate approximate token count (rough estimate: 1 token ≈ 4 chars)
    tokens = len(text) // 4
    
    return {
        "chunk_id": f"qa_chunk_{chunk_id}",
        "text": text,
        "metadata": metadata,
        "tokens": tokens,
        "source": "new_dataset_qa_pairs"
    }


def analyze_dataset(qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze QA pairs dataset for statistics."""
    stats = {
        "total_pairs": len(qa_pairs),
        "documents": Counter(qa["document"] for qa in qa_pairs),
        "types": Counter(qa.get("type", "unknown") for qa in qa_pairs),
        "difficulties": Counter(qa.get("difficulty", "medium") for qa in qa_pairs),
        "avg_question_length": sum(len(qa["question"]) for qa in qa_pairs) / len(qa_pairs),
        "avg_answer_length": sum(len(qa["answer"]) for qa in qa_pairs) / len(qa_pairs),
    }
    
    # Calculate average combined length
    avg_combined = stats["avg_question_length"] + stats["avg_answer_length"]
    stats["avg_combined_length"] = avg_combined
    stats["avg_tokens"] = int(avg_combined // 4)
    
    return stats


def print_statistics(stats: Dict[str, Any]):
    """Print dataset statistics."""
    print("\n" + "="*60)
    print("QA DATASET STATISTICS")
    print("="*60)
    
    print(f"\n📊 Total QA Pairs: {stats['total_pairs']:,}")
    print(f"📏 Average Chunk Size: {stats['avg_tokens']} tokens ({int(stats['avg_combined_length'])} chars)")
    
    print("\n📚 Documents:")
    for doc, count in stats['documents'].most_common():
        print(f"  • {doc}: {count} pairs")
    
    print("\n🏷️  Question Types:")
    for qtype, count in stats['types'].most_common():
        pct = (count / stats['total_pairs']) * 100
        print(f"  • {qtype}: {count} ({pct:.1f}%)")
    
    print("\n📈 Difficulty Distribution:")
    for diff, count in stats['difficulties'].most_common():
        pct = (count / stats['total_pairs']) * 100
        print(f"  • {diff}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*60)


def save_chunks(chunks: List[Dict[str, Any]], output_path: Path):
    """Save transformed chunks to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Saved {len(chunks):,} chunks to: {output_path}")


def main():
    """Main transformation pipeline."""
    # Paths
    repo_root = Path(__file__).parent.parent
    qa_dir = repo_root / "new_dataset" / "qa_pairs"
    output_path = repo_root / "outputs" / "chunks" / "qa_chunks.jsonl"
    
    # Override output path from command line if provided
    if len(sys.argv) > 1 and sys.argv[1] == "--output":
        output_path = Path(sys.argv[2])
    
    print("🔄 Starting QA Pairs Transformation...")
    print(f"📂 Input Directory: {qa_dir}")
    print(f"📄 Output File: {output_path}")
    
    # Load all QA pairs
    print("\n📥 Loading QA pairs...")
    qa_pairs = load_qa_pairs(qa_dir)
    
    if not qa_pairs:
        print("❌ No QA pairs found! Check the input directory.")
        return
    
    # Analyze dataset
    stats = analyze_dataset(qa_pairs)
    print_statistics(stats)
    
    # Transform to chunks
    print("\n🔧 Transforming QA pairs to DCE chunks...")
    chunks = []
    for idx, qa in enumerate(qa_pairs):
        chunk = transform_qa_to_chunk(qa, idx)
        chunks.append(chunk)
        
        # Progress indicator
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1:,} / {len(qa_pairs):,} pairs...")
    
    # Save chunks
    save_chunks(chunks, output_path)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ TRANSFORMATION COMPLETE")
    print("="*60)
    print(f"Input:  {len(qa_pairs):,} QA pairs")
    print(f"Output: {len(chunks):,} DCE chunks")
    print(f"Avg Size: {stats['avg_tokens']} tokens/chunk")
    print(f"File: {output_path}")
    print("\n💡 Next Steps:")
    print("  1. Run bias validation: python scripts/validate_qa_bias.py")
    print("  2. Upload to Pinecone: python scripts/upload_to_pinecone.py")
    print("  3. Test with DCE: python scripts/auto_rag_pipeline.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
