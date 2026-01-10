"""
Test biotech QA retrieval from Pinecone
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.vectordb import PineconeLTM
from src.embedding import HuggingFaceEmbedder
from src.config import Config

def main():
    print("="*60)
    print("  BIOTECH QA RETRIEVAL TEST")
    print("="*60)
    
    # Initialize
    embedder = HuggingFaceEmbedder(
        model="BAAI/bge-small-en-v1.5",
        api_key=Config.HF_TOKEN
    )
    
    ltm = PineconeLTM(
        api_key=Config.PINECONE_API_KEY,
        index_name="self-evolving-dce-ltm"
    )
    
    # Test queries
    queries = [
        "What is de-extinction?",
        "Explain CRISPR gene editing",
        "What is Colossal Biosciences working on?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        # Embed query
        query_vec = embedder.embed_query(query)
        
        # Retrieve
        results = ltm.query(
            query_vector=query_vec,
            namespace="qa_biotech",
            top_k=3
        )
        
        print(f"\nTop 3 Results:")
        for i, r in enumerate(results, 1):
            # Debug: print the actual structure
            if i == 1:
                print(f"\nDEBUG - Result type: {type(r)}")
                print(f"DEBUG - Result dir: {dir(r)[:10]}")
                print(f"DEBUG - Full result: {r}\n")
            
            score = r.get('score', 0) if isinstance(r, dict) else getattr(r, 'score', 0)
            metadata = r.get('metadata', {}) if isinstance(r, dict) else getattr(r, 'metadata', {})
            
            print(f"\n{i}. Score: {score:.3f}")
            
            # Try different metadata access methods
            text = None
            if isinstance(metadata, dict):
                text = metadata.get('text') or metadata.get('content') or metadata.get('answer')
            elif hasattr(metadata, '__dict__'):
                text = getattr(metadata, 'text', None) or getattr(metadata, 'content', None)
            
            if text:
                print(f"   Text: {text[:200]}...")
            else:
                print(f"   Text: N/A (metadata: {metadata})")
    
    print("\n" + "="*60)
    print("✅ Retrieval test complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
