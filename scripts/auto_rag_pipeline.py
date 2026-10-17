#!/usr/bin/env python3
"""
Auto-RAG Pipeline
Automated pipeline: Dataset -> Embeddings -> Vector DB -> Retrieval Ready

Usage:
    python scripts/auto_rag_pipeline.py --dataset data/my_docs.jsonl
    python scripts/auto_rag_pipeline.py --use-mdn  # Use existing MDN dataset

This script fulfills management's requirement:
1. Load dataset automatically
2. Generate embeddings and store in vector DB
3. Stand up retrieval pipeline
4. No human intervention required
"""
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add repo root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.ingestion.loaders import MDNLoader, DataLoader
from src.chunking.splitter import DocumentChunker
from src.embedding.embedder import HuggingFaceEmbedder
from src.vectordb.pinecone_ltm import PineconeLTM


def print_header(title: str):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def load_dataset(args) -> List[Dict[str, Any]]:
    """Load dataset from file or MDN repository"""
    print_header("Step 1: Load Dataset")
    
    if args.use_mdn:
        print("📥 Loading MDN documentation...")
        loader = MDNLoader()
        docs = loader.load_clean(
            output_path=Config.MVP_KB_OUTPUTS / "mdn_learn_docs_clean.jsonl"
        )
        print(f"✅ Loaded {len(docs)} MDN documents")
        return docs
    
    elif args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        print(f"📥 Loading dataset from: {dataset_path}")
        docs = DataLoader.load_jsonl(dataset_path)
        print(f"✅ Loaded {len(docs)} documents")
        return docs
    
    else:
        raise ValueError("Must specify either --dataset or --use-mdn")


def chunk_documents(docs: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    """Chunk documents into retrieval units"""
    print_header("Step 2: Chunk Documents")
    
    print(f"⚙️ Chunking settings:")
    print(f"  Max tokens: {args.max_chunk_tokens}")
    print(f"  Min tokens: {args.min_chunk_tokens}")
    
    chunker = DocumentChunker(
        max_chunk_tokens=args.max_chunk_tokens,
        min_chunk_tokens=args.min_chunk_tokens
    )
    
    chunks = chunker.chunk_documents(docs)
    
    # Get statistics
    stats = chunker.get_stats(chunks)
    print(f"\n📊 Chunking results:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Avg tokens per chunk: {stats['avg_tokens']:.1f}")
    print(f"  Chunks per document: {stats['chunks_per_doc']:.1f}")
    
    # Save chunks
    output_path = Config.OUTPUTS_DIR / "chunks" / "chunks.jsonl"
    DataLoader.save_jsonl(chunks, output_path)
    print(f"\n✅ Saved chunks to: {output_path}")
    
    return chunks


def generate_embeddings(
    chunks: List[Dict[str, Any]],
    args
) -> np.ndarray:
    """Generate embeddings for all chunks"""
    print_header("Step 3: Generate Embeddings")
    
    print(f"🤖 Embedding model: {args.embedding_model}")
    print(f"📦 Processing {len(chunks)} chunks...")
    
    # Initialize embedder
    embedder = HuggingFaceEmbedder(model=args.embedding_model)
    
    # Extract chunk texts
    chunk_texts = [c["text"] for c in chunks]
    
    # Generate embeddings in batches
    embeddings = embedder.embed_in_batches(
        chunk_texts,
        batch_size=args.batch_size,
        verbose=True
    )
    
    print(f"\n✅ Generated embeddings: {embeddings.shape}")
    
    # Save embeddings
    output_path = Config.OUTPUTS_DIR / "embeddings" / "chunk_embeddings.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"✅ Saved embeddings to: {output_path}")
    
    return embeddings


def store_in_vector_db(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    args
) -> PineconeLTM:
    """Store chunks and embeddings in Pinecone"""
    print_header("Step 4: Store in Vector Database")
    
    print(f"🔗 Connecting to Pinecone...")
    print(f"  Index: {args.index_name}")
    
    # Initialize Pinecone
    ltm = PineconeLTM(
        index_name=args.index_name,
        embedding_dim=embeddings.shape[1],
        metric="cosine"
    )
    
    # Check existing data
    stats_before = ltm.get_stats(namespace=ltm.NS_CHUNKS)
    print(f"  Existing vectors: {stats_before.get('vector_count', 0)}")
    
    # Upload chunks
    if args.skip_upload:
        print("⏭️ Skipping upload (--skip-upload flag)")
    else:
        print(f"\n📤 Uploading {len(chunks)} chunks...")
        ltm.upsert_chunks(chunks, embeddings, batch_size=200)
        
        stats_after = ltm.get_stats(namespace=ltm.NS_CHUNKS)
        print(f"\n✅ Vector DB updated")
        print(f"  Total vectors: {stats_after.get('vector_count', 0)}")
    
    return ltm


def test_retrieval(ltm: PineconeLTM, args):
    """Test retrieval with sample query"""
    print_header("Step 5: Test Retrieval")
    
    # Initialize embedder for query
    embedder = HuggingFaceEmbedder(model=args.embedding_model)
    
    # Test query
    query = args.test_query
    print(f"🔍 Test query: '{query}'")
    
    # Embed query
    query_vec = embedder.embed_query(query)
    
    # Search
    results = ltm.query(
        query_vector=query_vec,
        top_k=3,
        namespace=ltm.NS_CHUNKS
    )
    
    print(f"\n✅ Found {len(results)} results:\n")
    for i, match in enumerate(results, 1):
        metadata = match.get('metadata', {})
        score = match.get('score', 0)
        print(f"{i}. Score: {score:.4f}")
        print(f"   Title: {metadata.get('title', 'N/A')}")
        print(f"   Heading: {metadata.get('heading', 'N/A')}")
        print()


def print_summary(chunks: List[Dict[str, Any]], ltm: PineconeLTM):
    """Print final summary"""
    print_header("Pipeline Complete! 🎉")
    
    print("📊 Summary:")
    print(f"  ✅ Documents loaded and chunked")
    print(f"  ✅ {len(chunks)} chunks embedded")
    print(f"  ✅ Vector database ready")
    print(f"  ✅ Retrieval pipeline active")
    
    print(f"\n🔗 Pinecone Index: {ltm.index_name}")
    print(f"📦 Namespace: {ltm.NS_CHUNKS}")
    
    stats = ltm.get_stats(namespace=ltm.NS_CHUNKS)
    print(f"📈 Total vectors: {stats.get('vector_count', 0)}")
    
    print("\n✨ The retrieval system is ready for queries!")
    print("\nNext steps:")
    print("  1. Query the vector DB using src.vectordb.pinecone_ltm")
    print("  2. Build dynamic context composition (Notebook 04)")
    print("  3. Add evaluation and policy evolution")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-RAG Pipeline: Automated dataset -> embeddings -> vector DB"
    )
    
    # Dataset options
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset",
        type=str,
        help="Path to JSONL dataset file"
    )
    dataset_group.add_argument(
        "--use-mdn",
        action="store_true",
        help="Use MDN documentation dataset"
    )
    
    # Chunking options
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=300,
        help="Maximum tokens per chunk (default: 300)"
    )
    parser.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=80,
        help="Minimum tokens per chunk (default: 80)"
    )
    
    # Embedding options
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="HuggingFace embedding model (default: BAAI/bge-small-en-v1.5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding generation (default: 16)"
    )
    
    # Vector DB options
    parser.add_argument(
        "--index-name",
        type=str,
        default="self-evolving-dce-ltm",
        help="Pinecone index name (default: self-evolving-dce-ltm)"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading to Pinecone (for testing)"
    )
    
    # Test options
    parser.add_argument(
        "--test-query",
        type=str,
        default="Explain semantic HTML",
        help="Test query for retrieval (default: 'Explain semantic HTML')"
    )
    
    args = parser.parse_args()
    
    # Validate configuration
    Config.validate()
    Config.ensure_dirs()
    
    try:
        # Execute pipeline
        docs = load_dataset(args)
        chunks = chunk_documents(docs, args)
        embeddings = generate_embeddings(chunks, args)
        ltm = store_in_vector_db(chunks, embeddings, args)
        test_retrieval(ltm, args)
        print_summary(chunks, ltm)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
