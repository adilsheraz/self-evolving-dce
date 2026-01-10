"""
Test script to verify all DCE modules are properly importable

This validates the framework integration is complete and all modules
can be imported individually or through the main src package.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_individual_imports():
    """Test importing from individual modules"""
    print("🔍 Testing individual module imports...")
    
    try:
        from ingestion import DataLoader, MDNLoader
        print("  ✅ ingestion: DataLoader, MDNLoader")
    except ImportError as e:
        print(f"  ❌ ingestion: {e}")
        return False
    
    try:
        from chunking import DocumentChunker, estimate_tokens
        print("  ✅ chunking: DocumentChunker, estimate_tokens")
    except ImportError as e:
        print(f"  ❌ chunking: {e}")
        return False
    
    try:
        from embedding import EmbeddingProvider, HuggingFaceEmbedder
        print("  ✅ embedding: EmbeddingProvider, HuggingFaceEmbedder")
    except ImportError as e:
        print(f"  ❌ embedding: {e}")
        return False
    
    try:
        from context import ContextComposer, render_context_text
        print("  ✅ context: ContextComposer, render_context_text")
    except ImportError as e:
        print(f"  ❌ context: {e}")
        return False
    
    try:
        from evaluation import ContextEvaluator, EVAL_WEIGHTS
        print("  ✅ evaluation: ContextEvaluator, EVAL_WEIGHTS")
    except ImportError as e:
        print(f"  ❌ evaluation: {e}")
        return False
    
    try:
        from evolution import PolicyEvolver, build_policy_variant
        print("  ✅ evolution: PolicyEvolver, build_policy_variant")
    except ImportError as e:
        print(f"  ❌ evolution: {e}")
        return False
    
    try:
        from autoft import QAGenerator, QAValidator
        print("  ✅ autoft: QAGenerator, QAValidator")
    except ImportError as e:
        print(f"  ❌ autoft: {e}")
        return False
    
    try:
        from vectordb import PineconeLTM
        print("  ✅ vectordb: PineconeLTM")
    except ImportError as e:
        print(f"  ❌ vectordb: {e}")
        return False
    
    try:
        from letta import create_policy_strategist, LettaPolicyStrategist
        print("  ✅ letta: create_policy_strategist, LettaPolicyStrategist")
    except ImportError as e:
        print(f"  ❌ letta: {e}")
        return False
    
    try:
        # Note: orchestration requires package context, test separately
        print("  ⏭️  orchestration: (requires package context, tested in package imports)")
    except ImportError as e:
        print(f"  ❌ orchestration: {e}")
        return False
    
    try:
        from config import Config
        print("  ✅ config: Config")
    except ImportError as e:
        print(f"  ❌ config: {e}")
        return False
    
    return True


def test_src_package_imports():
    """Test importing from main src package"""
    print("\n🔍 Testing src package imports...")
    
    try:
        # Remove src from path to test package import
        sys.path.pop(0)
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        # Test comprehensive import
        from src import (
            Config,
            DataLoader, MDNLoader,
            DocumentChunker, estimate_tokens,
            EmbeddingProvider, HuggingFaceEmbedder,
            ContextComposer, render_context_text,
            ContextEvaluator, EVAL_WEIGHTS,
            PolicyEvolver, build_policy_variant,
            QAGenerator, QAValidator,
            PineconeLTM,
            create_policy_strategist, LettaPolicyStrategist,
            SelfEvolvingLoop
        )
        print("  ✅ All classes importable from 'src' package")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_version_info():
    """Test version and metadata"""
    print("\n🔍 Testing package metadata...")
    
    try:
        sys.path.pop(0)
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        import src
        print(f"  ✅ Version: {src.__version__}")
        print(f"  ✅ Author: {src.__author__}")
        return True
        
    except Exception as e:
        print(f"  ❌ Metadata check failed: {e}")
        return False


def test_framework_integration():
    """Test that modules can work together"""
    print("\n🔍 Testing framework integration...")
    
    try:
        sys.path.pop(0)
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from src import ContextComposer, ContextEvaluator
        import numpy as np
        
        # Create sample chunks
        chunks = [
            {
                "chunk_id": "test_1",
                "doc_id": "doc1",
                "title": "Test Doc",
                "text": "Sample text content for testing",
                "tokens": 10
            },
            {
                "chunk_id": "test_2",
                "doc_id": "doc1",
                "title": "Test Doc",
                "text": "Another sample content for diversity",
                "tokens": 10
            }
        ]
        
        # Create sample embeddings
        embeddings = np.random.randn(2, 384).astype(np.float32)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        
        # Test composer
        composer = ContextComposer(
            chunks=chunks,
            chunk_embeddings=embeddings
        )
        
        query_vec = np.random.randn(384).astype(np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        
        pkg = composer.compose(
            query="test query",
            candidate_idxs=[0, 1],
            query_vec=query_vec
        )
        
        # Test evaluator
        evaluator = ContextEvaluator()
        scores = evaluator.evaluate(pkg)
        
        print(f"  ✅ ContextComposer created package with {len(pkg['selected'])} chunks")
        print(f"  ✅ ContextEvaluator computed scores: {list(scores.keys())}")
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all import tests"""
    print("="*80)
    print("🧪 DCE Framework Import Validation")
    print("="*80)
    
    results = []
    
    # Run all tests
    results.append(("Individual Imports", test_individual_imports()))
    results.append(("Package Imports", test_src_package_imports()))
    results.append(("Version Info", test_version_info()))
    results.append(("Framework Integration", test_framework_integration()))
    
    # Summary
    print("\n" + "="*80)
    print("📊 Test Summary")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All imports working! Framework is ready for integration.")
        print("\nYou can now use:")
        print("  from src import ContextComposer, PineconeLTM, ...")
        print("  from src.letta import create_policy_strategist")
        print("  from src.orchestration import SelfEvolvingLoop")
    else:
        print("\n⚠️  Some imports failed. Review errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
