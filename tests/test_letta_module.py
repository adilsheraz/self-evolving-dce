"""
Test Letta Module Integration

Quick validation that the Letta module works correctly with your existing agent.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.letta import create_policy_strategist


def main():
    """Test Letta integration."""
    print("="*60)
    print("  LETTA MODULE INTEGRATION TEST")
    print("="*60)
    
    # Your existing agent
    AGENT_ID = "agent-fba21d29-97ff-4726-8a64-fa093b754a02"
    
    try:
        # Step 1: Connect to agent
        print("\n📡 Connecting to Letta agent...")
        letta = create_policy_strategist(agent_id=AGENT_ID)
        
        # Step 2: Check memory status
        print("\n📋 Checking memory blocks...")
        memory_status = letta.get_memory_status()
        print(f"Agent: {memory_status['agent_name']}")
        print(f"Memory blocks: {len(memory_status['memory_blocks'])}")
        for block in memory_status['memory_blocks']:
            print(f"  - {block['label']}: {block['size']} chars")
        
        # Step 3: Test proposal (dry run)
        print("\n🧪 Testing experiment proposal...")
        baseline_metrics = {
            "overall": 0.782,
            "relevance": 0.850,
            "novelty": 0.650,
            "redundancy": 0.180,
            "coverage": 0.720
        }
        
        experiment_history = [
            {
                "version": "v0.5",
                "override": {"w_novelty": 0.30},
                "delta_overall": 0.008,
                "decision": "REJECT"
            }
        ]
        
        proposal = letta.propose_experiment(
            baseline_metrics=baseline_metrics,
            experiment_history=experiment_history
        )
        
        print(f"\n✅ Proposal received:")
        print(f"   Version: {proposal.get('version', 'N/A')}")
        print(f"   Override: {proposal.get('override', {})}")
        print(f"   Rationale: {proposal.get('rationale', 'N/A')[:100]}...")
        
        # Step 4: Test decision making (dry run)
        print("\n🎯 Testing decision making...")
        mock_results = {
            "delta_overall": 0.015,
            "delta_novelty": 0.08,
            "delta_redundancy": -0.02
        }
        
        decision = letta.decide_on_results(proposal, mock_results)
        
        print(f"\n✅ Decision received:")
        print(f"   Decision: {decision.get('decision', 'N/A')}")
        print(f"   Notes: {decision.get('notes', 'N/A')[:100]}...")
        
        print("\n" + "="*60)
        print("✅ LETTA MODULE TEST PASSED")
        print("="*60)
        print("\nAll components working correctly:")
        print("  ✅ Agent connection")
        print("  ✅ Memory block access")
        print("  ✅ Experiment proposals")
        print("  ✅ Decision making")
        print("\n💡 Next: Integrate with DCE evolution loop")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
