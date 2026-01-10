"""
Upload existing chunks and embeddings to Pinecone.
Use this when embeddings are already generated but upload step failed.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.vectordb.pinecone_ltm import PineconeLTM


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    """Load embeddings from numpy file."""
    return np.load(embeddings_path)


def upload_to_pinecone(
    chunks: list[dict],
    embeddings: np.ndarray,
    index_name: str,
    namespace: str
):
    """Upload chunks and embeddings to Pinecone."""
    print(f"\n{'='*60}")
    print(f"  Uploading to Pinecone")
    print(f"{'='*60}\n")
    
    print(f"🔗 Connecting to Pinecone...")
    print(f"  Index: {index_name}")
    print(f"  Namespace: {namespace}")
    
    # Initialize Pinecone LTM
    ltm = PineconeLTM(
        index_name=index_name,
        embedding_dim=embeddings.shape[1],
        metric="cosine"
    )
    
    print(f"\n📤 Uploading {len(chunks)} vectors...")
    
    # Upload in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        
        ltm.upsert_chunks(
            chunks=batch_chunks,
            embeddings=batch_embeddings,
            namespace=namespace
        )
        
        print(f"  ✅ Uploaded batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({i+len(batch_chunks)}/{len(chunks)} vectors)")
    
    print(f"\n✅ Successfully uploaded {len(chunks)} vectors to Pinecone!")
    
    # Test retrieval
    print(f"\n{'='*60}")
    print(f"  Testing Retrieval")
    print(f"{'='*60}\n")
    
    test_query = "What is semantic HTML?"
    print(f"🔍 Query: '{test_query}'")
    
    results = ltm.query(test_query, namespace=namespace, top_k=3)
    
    print(f"\n📊 Top {len(results)} results:\n")
    for i, res in enumerate(results, 1):
        print(f"{i}. [Score: {res['score']:.4f}]")
        print(f"   Source: {res['metadata'].get('source', 'N/A')}")
        print(f"   Text: {res['text'][:150]}...\n")
    
    return ltm


def main():
    parser = argparse.ArgumentParser(
        description="Upload existing chunks and embeddings to Pinecone"
    )
    parser.add_argument(
        "--chunks",
        default="outputs/chunks/chunks.jsonl",
        help="Path to chunks JSONL file (default: outputs/chunks/chunks.jsonl)"
    )
    parser.add_argument(
        "--embeddings",
        default="outputs/embeddings/chunk_embeddings.npy",
        help="Path to embeddings NPY file (default: outputs/embeddings/chunk_embeddings.npy)"
    )
    parser.add_argument(
        "--index-name",
        default="self-evolving-dce-ltm",
        help="Pinecone index name (default: self-evolving-dce-ltm)"
    )
    parser.add_argument(
        "--namespace",
        default="kb_chunks",
        help="Pinecone namespace (default: kb_chunks)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    config = Config()
    chunks_path = repo_root / args.chunks
    embeddings_path = repo_root / args.embeddings
    
    # Verify files exist
    if not chunks_path.exists():
        print(f"❌ Chunks file not found: {chunks_path}")
        sys.exit(1)
    
    if not embeddings_path.exists():
        print(f"❌ Embeddings file not found: {embeddings_path}")
        sys.exit(1)
    
    print(f"📂 Loading data...")
    print(f"  Chunks: {chunks_path}")
    print(f"  Embeddings: {embeddings_path}")
    
    # Load data
    chunks = load_chunks(chunks_path)
    embeddings = load_embeddings(embeddings_path)
    
    print(f"\n✅ Loaded {len(chunks)} chunks")
    print(f"✅ Loaded {len(embeddings)} embeddings (shape: {embeddings.shape})")
    
    # Verify alignment
    if len(chunks) != len(embeddings):
        print(f"\n❌ ERROR: Chunk count ({len(chunks)}) != Embedding count ({len(embeddings)})")
        sys.exit(1)
    
    try:
        # Upload to Pinecone
        upload_to_pinecone(chunks, embeddings, args.index_name, args.namespace)
        
        print(f"\n{'='*60}")
        print(f"✅ Upload completed successfully!")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
