"""
DCE Framework Integration Example

This demonstrates how to import and use DCE modules in external projects.
Shows both individual module usage and full framework integration.
"""

import sys
from pathlib import Path

# Add src to path (in real usage, install as package)
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Example 1: Import individual modules
# ============================================================================
print("=" * 80)
print("Example 1: Individual Module Usage")
print("=" * 80)

from src.context import ContextComposer
from src.evaluation import ContextEvaluator
from src.vectordb import PineconeLTM
from src.letta import create_policy_strategist

print("✅ Imported: ContextComposer, ContextEvaluator, PineconeLTM")
print("✅ Imported: create_policy_strategist")

# ============================================================================
# Example 2: Import from main package
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Package-Level Import")
print("=" * 80)

from src import (
    Config,
    DataLoader,
    DocumentChunker,
    HuggingFaceEmbedder,
    ContextComposer as Composer,  # Can rename
    PolicyEvolver,
    QAGenerator,
    PineconeLTM as VectorDB,  # Alias
)

import src

print("✅ Imported all core classes from 'src' package")
print(f"✅ DCE Version: {src.__version__}")

# ============================================================================
# Example 3: Use in external framework
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Integration in External Framework")
print("=" * 80)

class MyAIFramework:
    """Example external framework using DCE"""
    
    def __init__(self):
        # Import DCE components
        from src import ContextComposer, ContextEvaluator
        from src.letta import create_policy_strategist
        
        self.composer = None  # Will be created per query
        self.evaluator = ContextEvaluator()
        self.letta_agent = None  # Can create when needed
        
        print("✅ MyAIFramework initialized with DCE components")
    
    def process_query(self, query: str):
        """Example processing pipeline"""
        print(f"\n📝 Processing query: '{query}'")
        print("   1. Retrieve candidates (your vector DB)")
        print("   2. Create ContextComposer with candidates")
        print("   3. Compose optimal context package")
        print("   4. Evaluate with ContextEvaluator")
        print("   5. Return to LLM for generation")
        
        return {"status": "simulated", "query": query}

# Test the framework
framework = MyAIFramework()
result = framework.process_query("What is de-extinction?")
print(f"✅ Result: {result}")

# ============================================================================
# Example 4: Use orchestration for autonomous evolution
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Autonomous Evolution with Orchestration")
print("=" * 80)

from src.orchestration import SelfEvolvingLoop

print("✅ Imported SelfEvolvingLoop for autonomous policy evolution")
print("\nTo use:")
print("  loop = SelfEvolvingLoop(")
print("      chunks=your_chunks,")
print("      embeddings=your_embeddings,")
print("      letta_agent=your_agent")
print("  )")
print("  loop.run_evolution_cycle(baseline_queries)")

# ============================================================================
# Example 5: Complete DCE pipeline
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Complete DCE Pipeline")
print("=" * 80)

from src import (
    DataLoader,
    DocumentChunker,
    HuggingFaceEmbedder,
    ContextComposer,
    ContextEvaluator,
    PineconeLTM,
)

print("\n📋 Complete pipeline imports:")
print("  1. DataLoader - Ingest documents")
print("  2. DocumentChunker - Split into chunks")  
print("  3. HuggingFaceEmbedder - Generate embeddings")
print("  4. PineconeLTM - Store in vector DB")
print("  5. ContextComposer - Dynamic composition")
print("  6. ContextEvaluator - Measure quality")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("🎉 Framework Integration Complete!")
print("=" * 80)

print("\n📦 Available Import Patterns:")
print("\n1. Individual modules:")
print("   from src.context import ContextComposer")
print("   from src.letta import create_policy_strategist")

print("\n2. Package-level imports:")
print("   from src import ContextComposer, PineconeLTM, PolicyEvolver")

print("\n3. Full orchestration:")
print("   from src.orchestration import SelfEvolvingLoop")

print("\n4. Configuration:")
print("   from src import Config")

print("\n✅ All DCE components are properly packaged and ready for integration!")
print("✅ Can be imported into any Python project")
print("✅ Use in FastAPI, LangChain, LlamaIndex, or custom frameworks")
