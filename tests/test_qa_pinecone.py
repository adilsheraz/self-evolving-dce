"""
Generate embeddings for QA chunks and upload to Pinecone.
Quick integration test for the new biotech dataset.
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.embedding.embedder import HuggingFaceEmbedder
from src.vectordb.pinecone_ltm import PineconeLTM


def load_qa_chunks(chunks_path: Path) -> list[dict]:
    """Load QA chunks from JSONL file."""
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def generate_embeddings(chunks: list[dict], batch_size: int = 100) -> np.ndarray:
    """Generate embeddings for QA chunks."""
    print(f"\n🔧 Initializing embedder...")
    embedder = HuggingFaceEmbedder(model="BAAI/bge-small-en-v1.5")
    
    print(f"📊 Generating embeddings for {len(chunks)} chunks...")
    
    all_embeddings = []
    texts = [chunk["text"] for chunk in chunks]
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embedder.embed_texts(batch_texts)
        all_embeddings.extend(batch_embeddings)
    
    embeddings_array = np.array(all_embeddings)
    print(f"✅ Generated embeddings: {embeddings_array.shape}")
    
    return embeddings_array


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
    for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading batches"):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        
        ltm.upsert_chunks(
            chunks=batch_chunks,
            embeddings=batch_embeddings,
            namespace=namespace
        )
    
    print(f"\n✅ Successfully uploaded {len(chunks)} vectors to Pinecone!")
    
    return ltm


def test_retrieval(ltm: PineconeLTM, namespace: str):
    """Test retrieval with sample biotech queries."""
    print(f"\n{'='*60}")
    print(f"  Testing Retrieval")
    print(f"{'='*60}\n")
    
    test_queries = [
        "What is CRISPR-Cas9?",
        "How does de-extinction work?",
        "What are the ethical implications of genetic engineering?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = ltm.query(query, namespace=namespace, top_k=3)
        
        print(f"📊 Top {len(results)} results:\n")
        for i, res in enumerate(results, 1):
            qa_type = res['metadata'].get('qa_type', 'N/A')
            difficulty = res['metadata'].get('difficulty', 'N/A')
            print(f"{i}. [Score: {res['score']:.4f}] [{qa_type}/{difficulty}]")
            print(f"   {res['text'][:200]}...\n")


def main():
    """Main pipeline: Load QA chunks → Generate embeddings → Upload → Test."""
    
    print("="*60)
    print("  QA DATASET PINECONE INTEGRATION")
    print("="*60)
    
    # Paths
    chunks_path = repo_root / "outputs" / "chunks" / "qa_chunks.jsonl"
    embeddings_path = repo_root / "outputs" / "embeddings" / "qa_embeddings.npy"
    
    # Config
    index_name = "self-evolving-dce-ltm"
    namespace = "qa_biotech"
    
    # Step 1: Load chunks
    print(f"\n📂 Loading QA chunks from: {chunks_path}")
    if not chunks_path.exists():
        print(f"❌ Chunks file not found: {chunks_path}")
        print("💡 Run transformation first: python scripts/transform_qa_to_chunks.py")
        sys.exit(1)
    
    chunks = load_qa_chunks(chunks_path)
    print(f"✅ Loaded {len(chunks):,} chunks")
    
    # Step 2: Load or generate embeddings
    if embeddings_path.exists():
        print(f"\n📥 Loading existing embeddings from: {embeddings_path}")
        embeddings = np.load(embeddings_path)
        print(f"✅ Loaded embeddings: {embeddings.shape}")
    else:
        print(f"\n🎯 Generating embeddings...")
        embeddings = generate_embeddings(chunks, batch_size=100)
        
        # Save embeddings
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, embeddings)
        print(f"💾 Saved embeddings to: {embeddings_path}")
    
    # Step 3: Upload to Pinecone
    try:
        ltm = upload_to_pinecone(chunks, embeddings, index_name, namespace)
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Test retrieval
    test_retrieval(ltm, namespace)
    
    # Final summary
    print(f"\n{'='*60}")
    print("✅ QA DATASET INTEGRATION COMPLETE")
    print("="*60)
    print(f"Chunks: {len(chunks):,}")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Index: {index_name}")
    print(f"Namespace: {namespace}")
    print(f"\n💡 Next: Test with DCE context selection")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
