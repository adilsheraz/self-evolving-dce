"""
Demo: Self-Evolving DCE System

This script demonstrates the autonomous policy evolution loop that combines:
- Letta Policy Strategist (strategic decision-making)
- DCE Engine (deterministic execution)
- Autonomous improvement (self-evolving)

The system will:
1. Evaluate baseline policy
2. Ask Letta agent for next experiment
3. Execute experiment with DCE
4. Letta decides ACCEPT/REJECT
5. Update policy if accepted
6. Repeat until convergence
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.orchestration import SelfEvolvingLoop


def main():
    """Run self-evolving demo."""
    
    print("="*70)
    print("  SELF-EVOLVING DCE DEMO")
    print("="*70)
    print("\nThis demo shows autonomous policy evolution using Letta agent.")
    print("The system will run experiments and improve itself automatically.")
    print("\n" + "="*70)
    
    # Configuration
    BASE_POLICY = {
        "w_novelty": 0.20,
        "w_redundancy": -0.30,
        "w_doc_diversity": 0.20,
        "w_retrieval": 0.60,
        "budget_tokens": 1500,
        "min_chunk_chars": 200,
        "policy_version": "v0.1"
    }
    
    LETTA_AGENT_ID = "agent-fba21d29-97ff-4726-8a64-fa093b754a02"
    PINECONE_INDEX = "dce-knowledge-base"
    
    # Test queries (using biotech domain from new dataset)
    EVAL_QUERIES = [
        "What is de-extinction and how does it work?",
        "Explain CRISPR gene editing technology",
        "What is Colossal Biosciences working on?",
        "How does genetic engineering help conservation?",
        "What are the ethical concerns with de-extinction?"
    ]
    
    try:
        # Initialize self-evolving loop
        print("\n🚀 Initializing Self-Evolving Loop...")
        
        loop = SelfEvolvingLoop(
            base_policy=BASE_POLICY,
            letta_agent_id=LETTA_AGENT_ID,
            pinecone_index_name=PINECONE_INDEX,
            eval_queries=EVAL_QUERIES,
            state_dir=Path("state/evolution_demo")
        )
        
        print("✅ Initialized successfully")
        print(f"   Letta Agent: {loop.letta.agent_name}")
        print(f"   Base Policy: {BASE_POLICY['policy_version']}")
        print(f"   Eval Queries: {len(EVAL_QUERIES)}")
        
        # Run evolution cycle
        print("\n" + "="*70)
        print("  STARTING AUTONOMOUS EVOLUTION")
        print("="*70)
        print("\nThe system will now:")
        print("  1. Evaluate baseline policy")
        print("  2. Letta proposes experiments")
        print("  3. Execute and evaluate each proposal")
        print("  4. Letta decides accept/reject")
        print("  5. Update policy if improvement found")
        print("  6. Repeat until convergence or max iterations")
        print("\n" + "="*70)
        
        summary = loop.run_evolution_cycle(
            max_iterations=10,
            convergence_threshold=0.005
        )
        
        # Final report
        print("\n" + "="*70)
        print("  FINAL REPORT")
        print("="*70)
        
        print(f"\n📊 Evolution Statistics:")
        print(f"   Total Iterations: {summary['iterations']}")
        print(f"   Experiments Accepted: {summary['accepted']}")
        print(f"   Experiments Rejected: {summary['rejected']}")
        print(f"   Success Rate: {summary['accepted']/summary['iterations']*100:.1f}%")
        
        print(f"\n🎯 Final Policy ({summary['final_version']}):")
        for param, value in summary['final_policy'].items():
            print(f"   {param:20s}: {value}")
        
        print("\n" + "="*70)
        print("✅ DEMO COMPLETE")
        print("="*70)
        
        print("\n💡 What just happened:")
        print("   • Your Letta agent autonomously proposed policy experiments")
        print("   • DCE executed each experiment and measured results")
        print("   • Letta analyzed results and decided accept/reject")
        print("   • System evolved to improved policy automatically")
        print("   • No manual tuning required!")
        
        print(f"\n📁 Check state/evolution_demo/ for:")
        print("   • baseline_metrics.json - Initial performance")
        print("   • policy_v*.json - Each accepted policy version")
        print("   • evolution_history.json - Complete audit trail")
        
        print("\n" + "="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        print("   Progress saved to state/evolution_demo/")
        
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
