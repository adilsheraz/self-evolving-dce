"""
Quick Test: Self-Evolving DCE with Letta Agent

This is a simplified test that demonstrates Letta's autonomous decision-making
without requiring full DCE integration.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.letta import create_policy_strategist
import json


def simulate_baseline():
    """Simulate baseline metrics."""
    return {
        "overall": 0.782,
        "relevance": 0.850,
        "novelty": 0.650,
        "redundancy": 0.180,
        "coverage": 0.720
    }


def simulate_experiment(baseline_metrics, override):
    """
    Simulate running an experiment with policy override.
    For demo purposes, we'll use simple heuristics.
    """
    import random
    random.seed(42)
    
    # Simulate some improvement based on the override
    deltas = {}
    for key in baseline_metrics:
        if key in ["novelty", "coverage"]:
            # Positive changes more likely for these
            deltas[f"delta_{key}"] = random.uniform(0.01, 0.05)
        elif key == "redundancy":
            # Negative change (reduction) is good
            deltas[f"delta_{key}"] = random.uniform(-0.03, -0.01)
        else:
            # Small variations
            deltas[f"delta_{key}"] = random.uniform(-0.01, 0.02)
    
    return deltas


def main():
    """Run simplified self-evolving demo."""
    
    print("="*70)
    print("  SIMPLIFIED SELF-EVOLVING DEMO")
    print("="*70)
    print("\nThis demo shows Letta agent's autonomous decision-making.")
    print("(Using simulated DCE execution for simplicity)")
    print("\n" + "="*70)
    
    # Configuration
    BASE_POLICY = {
        "w_novelty": 0.20,
        "w_redundancy": -0.30,
        "w_doc_diversity": 0.20,
        "w_retrieval": 0.60,
        "policy_version": "v0.1"
    }
    
    LETTA_AGENT_ID = "agent-fba21d29-97ff-4726-8a64-fa093b754a02"
    
    try:
        # Initialize Letta agent
        print("\n🚀 Connecting to Letta Policy Strategist...")
        letta = create_policy_strategist(agent_id=LETTA_AGENT_ID)
        print(f"✅ Connected to agent: {letta.agent_name}")
        
        # Baseline
        print("\n" + "="*70)
        print("  BASELINE EVALUATION")
        print("="*70)
        baseline_metrics = simulate_baseline()
        print("\n📊 Baseline Metrics:")
        for metric, value in baseline_metrics.items():
            print(f"   {metric:12s}: {value:.3f}")
        
        # Run 3 iterations
        experiment_history = []
        current_policy = BASE_POLICY.copy()
        
        for iteration in range(1, 4):
            print(f"\n{'='*70}")
            print(f"  ITERATION {iteration}")
            print(f"{'='*70}")
            
            # Step 1: Propose
            print("\n🤖 Asking Letta for next experiment...")
            proposal = letta.propose_experiment(
                baseline_metrics=baseline_metrics,
                experiment_history=experiment_history
            )
            
            print(f"\n✅ Proposal received:")
            print(f"   Version: {proposal.get('version', 'N/A')}")
            print(f"   Override: {proposal.get('override', {})}")
            print(f"   Rationale: {proposal.get('rationale', 'N/A')[:150]}...")
            
            # Step 2: Execute (simulated)
            print(f"\n🧪 Executing experiment (simulated)...")
            deltas = simulate_experiment(baseline_metrics, proposal['override'])
            
            print(f"\n📈 Results:")
            for metric, delta in deltas.items():
                sign = "+" if delta >= 0 else ""
                print(f"   {metric:18s}: {sign}{delta:.4f}")
            
            # Step 3: Decide
            print("\n🎯 Asking Letta to decide...")
            decision = letta.decide_on_results(proposal, deltas)
            
            print(f"\n✅ Decision: {decision.get('decision', 'N/A')}")
            print(f"   Notes: {decision.get('notes', 'N/A')[:150]}...")
            
            # Step 4: Update
            if decision['decision'] == "ACCEPT":
                print("\n✅ ACCEPTED - Updating policy")
                current_policy.update(proposal['override'])
                current_policy['policy_version'] = proposal['version']
                
                # Update baseline for next iteration
                for key in baseline_metrics:
                    if f"delta_{key}" in deltas:
                        baseline_metrics[key] += deltas[f"delta_{key}"]
                
                # Update Letta memory
                letta.update_experiment_memory({
                    "version": proposal['version'],
                    "override": proposal['override'],
                    "results": deltas,
                    "decision": "ACCEPT",
                    "notes": decision.get('notes', '')
                })
            else:
                print("\n❌ REJECTED - Policy unchanged")
                letta.update_experiment_memory({
                    "version": proposal['version'],
                    "override": proposal['override'],
                    "results": deltas,
                    "decision": "REJECT",
                    "notes": decision.get('notes', '')
                })
            
            # Track history
            experiment_history.append({
                "version": proposal['version'],
                "override": proposal['override'],
                "delta_overall": deltas.get('delta_overall', 0),
                "decision": decision['decision']
            })
        
        # Summary
        print("\n" + "="*70)
        print("  DEMO COMPLETE")
        print("="*70)
        
        accepted = sum(1 for h in experiment_history if h['decision'] == 'ACCEPT')
        rejected = len(experiment_history) - accepted
        
        print(f"\n📊 Summary:")
        print(f"   Total Iterations: {len(experiment_history)}")
        print(f"   Accepted: {accepted}")
        print(f"   Rejected: {rejected}")
        print(f"   Final Policy Version: {current_policy['policy_version']}")
        
        print(f"\n🎯 Final Policy:")
        for param, value in current_policy.items():
            print(f"   {param:20s}: {value}")
        
        print("\n💡 What just happened:")
        print("   • Letta agent proposed 3 policy experiments")
        print("   • Each experiment was 'executed' (simulated)")
        print("   • Letta analyzed results and decided accept/reject")
        print("   • Policy evolved based on Letta's decisions")
        print("   • All decisions stored in agent's memory")
        
        print("\n🔍 Check Letta agent's memory:")
        memory = letta.get_memory_status()
        print(f"   Agent: {memory['agent_name']}")
        print(f"   Memory blocks: {len(memory['memory_blocks'])}")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
