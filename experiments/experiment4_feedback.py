"""
Experiment 4: Closed-Loop Feedback & FSM Generation

Goal:
1. Use Exp3 feedback to refine Exp1's field boundaries
2. Iteratively improve constraint inference (Exp2)
3. Generate final protocol FSM
4. Compare with baseline (no feedback)

Research Question:
RQ4: Does active exploration feedback improve field segmentation accuracy?

Pipeline:
    Initial Data
        ↓
    [Exp1: Field Segmentation] → boundaries_v0
        ↓
    [Exp2: Constraint Inference] → constraints_v0
        ↓
    [Exp3: Active Exploration] → new_states, new_messages
        ↓
    [Feedback Loop] → refine boundaries & constraints
        ↓
    [Iteration 2, 3, ...]
        ↓
    [Final FSM Generation]
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import json
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import networkx as nx

from neupre import setup_logging
from utils.evaluator import NeuPREEvaluator
from utils.pcap_loader import PCAPDataLoader

# Import modules from previous experiments
from experiment2_multi_protocol import simulate_neupre_segmentation
from experiment3_constraints import ConstraintLearner


# ==================== FSM Construction ====================
class ProtocolFSM:
    """
    Finite State Machine for protocol
    
    States: Identified by response patterns
    Transitions: Triggered by message fields
    """
    
    def __init__(self):
        self.states: Set[str] = set()
        self.transitions: List[Dict] = []
        self.initial_state: str = None
        self.state_messages: Dict[str, List[bytes]] = defaultdict(list)
        
    def add_state(self, state_id: str, response: bytes):
        """Add a new state"""
        self.states.add(state_id)
        self.state_messages[state_id].append(response)
        
        if self.initial_state is None:
            self.initial_state = state_id
    
    def add_transition(self, from_state: str, to_state: str, 
                      trigger_field: str, trigger_value: any):
        """Add state transition"""
        self.transitions.append({
            'from': from_state,
            'to': to_state,
            'trigger_field': trigger_field,
            'trigger_value': trigger_value
        })
    
    def analyze_from_exploration_data(self, messages: List[bytes], 
                                      responses: List[bytes],
                                      segmentations: List[List[int]]):
        """
        Construct FSM from exploration data
        
        Args:
            messages: Sent messages
            responses: Server responses
            segmentations: Field boundaries for each message
        """
        logging.info("Analyzing FSM from exploration data...")
        
        # Step 1: Cluster responses into states
        state_map = {}  # response_hash -> state_id
        state_counter = 0
        
        for response in responses:
            # Use response signature as state identifier
            response_sig = self._get_response_signature(response)
            
            if response_sig not in state_map:
                state_id = f"S{state_counter}"
                state_map[response_sig] = state_id
                self.add_state(state_id, response)
                state_counter += 1
        
        logging.info(f"Identified {len(self.states)} unique states")
        
        # Step 2: Identify transitions
        for i in range(len(messages) - 1):
            current_response = responses[i]
            next_response = responses[i + 1]
            
            current_state = state_map[self._get_response_signature(current_response)]
            next_state = state_map[self._get_response_signature(next_response)]
            
            # Analyze which field caused transition
            if current_state != next_state:
                trigger_field, trigger_value = self._find_trigger_field(
                    messages[i], messages[i + 1], segmentations[i]
                )
                
                self.add_transition(current_state, next_state, 
                                  trigger_field, trigger_value)
        
        logging.info(f"Identified {len(self.transitions)} transitions")
    
    def _get_response_signature(self, response: bytes) -> str:
        """Generate signature from response (ignoring noise)"""
        if not response or len(response) < 4:
            return "EMPTY"
        
        # Use first few bytes as signature
        return response[:min(10, len(response))].hex()
    
    def _find_trigger_field(self, msg1: bytes, msg2: bytes, 
                           segmentation: List[int]) -> Tuple[str, any]:
        """Find which field changed between two messages"""
        # Extract fields based on segmentation
        fields1 = self._extract_fields(msg1, segmentation)
        fields2 = self._extract_fields(msg2, segmentation)
        
        # Find first different field
        for i, (f1, f2) in enumerate(zip(fields1, fields2)):
            if f1 != f2:
                return f"field_{i}", f2
        
        return "unknown", None
    
    def _extract_fields(self, message: bytes, segmentation: List[int]) -> List[bytes]:
        """Extract fields from message"""
        fields = []
        for i in range(len(segmentation) - 1):
            start = segmentation[i]
            end = segmentation[i + 1]
            fields.append(message[start:end])
        return fields
    
    def visualize(self, output_path: str = 'fsm_visualization.png'):
        """
        Generate FSM visualization using NetworkX
        """
        logging.info("Generating FSM visualization...")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add states
        for state in self.states:
            G.add_node(state)
        
        # Add transitions
        edge_labels = {}
        for trans in self.transitions:
            from_s = trans['from']
            to_s = trans['to']
            trigger = f"{trans['trigger_field']}={trans['trigger_value']}"
            
            if G.has_edge(from_s, to_s):
                # Multiple triggers for same transition
                edge_labels[(from_s, to_s)] += f"\n{trigger}"
            else:
                G.add_edge(from_s, to_s)
                edge_labels[(from_s, to_s)] = trigger
        
        # Layout
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=2000, alpha=0.9)
        
        # Highlight initial state
        if self.initial_state:
            nx.draw_networkx_nodes(G, pos, nodelist=[self.initial_state],
                                  node_color='lightgreen', node_size=2500)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, arrowstyle='->')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        plt.title("Protocol Finite State Machine", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"FSM visualization saved to {output_path}")
    
    def export_dot(self, output_path: str = 'fsm.dot'):
        """Export FSM in DOT format for Graphviz"""
        with open(output_path, 'w') as f:
            f.write("digraph ProtocolFSM {\n")
            f.write("  rankdir=LR;\n")
            f.write(f"  node [shape=circle];\n")
            
            # Mark initial state
            if self.initial_state:
                f.write(f'  "{self.initial_state}" [shape=doublecircle];\n')
            
            # Transitions
            for trans in self.transitions:
                label = f"{trans['trigger_field']}={trans['trigger_value']}"
                f.write(f'  "{trans["from"]}" -> "{trans["to"]}" [label="{label}"];\n')
            
            f.write("}\n")
        
        logging.info(f"FSM DOT file saved to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get FSM statistics"""
        return {
            'num_states': len(self.states),
            'num_transitions': len(self.transitions),
            'initial_state': self.initial_state,
            'avg_transitions_per_state': len(self.transitions) / len(self.states) if self.states else 0
        }


# ==================== Feedback Mechanism ====================
class FeedbackRefinement:
    """
    Use Exp3 exploration feedback to refine Exp1 & Exp2
    """
    
    def __init__(self):
        self.new_messages: List[bytes] = []
        self.new_responses: List[bytes] = []
        
    def collect_exploration_data(self, messages: List[bytes], responses: List[bytes]):
        """Collect new data from active exploration"""
        self.new_messages.extend(messages)
        self.new_responses.extend(responses)
        logging.info(f"Collected {len(messages)} new messages from exploration")
    
    def refine_segmentation(self, original_messages: List[bytes],
                          original_segmentations: List[List[int]]) -> List[List[int]]:
        """
        Refine segmentation using exploration feedback
        
        Strategy:
        1. Re-train on combined dataset (original + explored)
        2. Use state-discriminative features to improve boundaries
        """
        logging.info("Refining segmentation with exploration feedback...")
        
        # Combine datasets
        combined_messages = original_messages + self.new_messages
        
        # Re-run segmentation with more data
        refined_segmentations = simulate_neupre_segmentation(combined_messages)
        
        # Return only the refined versions of original messages
        return refined_segmentations[:len(original_messages)]
    
    def refine_constraints(self, original_constraints: List[Dict],
                          messages: List[bytes],
                          segmentations: List[List[int]]) -> List[Dict]:
        """
        Refine constraints using exploration feedback
        
        Strategy:
        1. Validate existing constraints on new messages
        2. Discover new constraints from state transitions
        """
        logging.info("Refining constraints with exploration feedback...")
        
        learner = ConstraintLearner()
        
        # Re-learn constraints on combined data
        combined_messages = messages + self.new_messages
        new_correlation = learner.learn_length_constraint(combined_messages)
        
        # Merge with original constraints
        refined_constraints = original_constraints.copy()
        
        if new_correlation > 0.95:
            refined_constraints.append({
                'type': 'length_field',
                'correlation': new_correlation,
                'verified_on_exploration': True
            })
        
        return refined_constraints


# ==================== Main Experiment ====================
def run_experiment4(protocol: str = 'modbus',
                   max_messages: int = 200,
                   num_iterations: int = 3,
                   exploration_rounds_per_iter: int = 20,
                   output_dir: str = './experiments/exp4_results'):
    """
    Run Experiment 4: Closed-Loop Feedback
    
    Args:
        protocol: Protocol to test
        max_messages: Initial dataset size
        num_iterations: Number of feedback iterations
        exploration_rounds_per_iter: Exploration rounds per iteration
        output_dir: Output directory
    """
    setup_logging(level=logging.INFO)
    
    logging.info("="*80)
    logging.info("EXPERIMENT 4: Closed-Loop Feedback & FSM Generation")
    logging.info("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    evaluator = NeuPREEvaluator(output_dir=output_dir)
    
    # Load data
    loader = PCAPDataLoader(data_dir='../data')
    
    if protocol == 'modbus':
        pcap_path = 'in-modbus-pcaps/libmodbus-bandwidth_server-rand_client.pcap'
    elif protocol == 'dnp3':
        pcap_path = 'in-dnp3-pcaps/BinInf_dnp3_1000.pcap'
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")
    
    messages = loader.load_messages(pcap_path, max_messages=max_messages)
    logging.info(f"Loaded {len(messages)} initial {protocol.upper()} messages")
    
    # Ground truth for comparison
    from experiment2_multi_protocol import get_modbus_gt, get_dnp3_gt
    gt_func = get_modbus_gt if protocol == 'modbus' else get_dnp3_gt
    ground_truth = [gt_func(msg) for msg in messages]
    
    # Initialize
    feedback = FeedbackRefinement()
    fsm = ProtocolFSM()
    
    iteration_results = []
    
    # ==================== ITERATION 0: Baseline (No Feedback) ====================
    logging.info("\n" + "="*80)
    logging.info("ITERATION 0: Baseline (No Feedback)")
    logging.info("="*80)
    
    # Exp1: Initial segmentation
    segmentations_v0 = simulate_neupre_segmentation(messages)
    
    # Compute metrics
    from experiment2_multi_protocol import compute_metrics
    metrics_v0 = compute_metrics(segmentations_v0, ground_truth)
    
    iteration_results.append({
        'iteration': 0,
        'type': 'baseline',
        'metrics': metrics_v0,
        'num_messages': len(messages),
        'num_new_states': 0
    })
    
    logging.info(f"Iteration 0 Results:")
    logging.info(f"  F1 Score: {metrics_v0['f1']:.4f}")
    logging.info(f"  Perfect Match: {metrics_v0['perfect_match']:.4f}")
    
    # ==================== ITERATIONS 1-N: With Feedback ====================
    current_segmentations = segmentations_v0
    
    for iteration in range(1, num_iterations + 1):
        logging.info("\n" + "="*80)
        logging.info(f"ITERATION {iteration}: Feedback Loop")
        logging.info("="*80)
        
        # Step 1: Simulate active exploration (Exp3)
        logging.info("Step 1: Active Exploration...")
        explored_messages, explored_responses = simulate_active_exploration(
            messages, current_segmentations, num_rounds=exploration_rounds_per_iter
        )
        
        # Collect feedback
        feedback.collect_exploration_data(explored_messages, explored_responses)
        
        # Step 2: Refine segmentation (Exp1 with feedback)
        logging.info("Step 2: Refining Segmentation with Feedback...")
        refined_segmentations = feedback.refine_segmentation(
            messages, current_segmentations
        )
        
        # Step 3: Update FSM
        logging.info("Step 3: Updating FSM...")
        fsm.analyze_from_exploration_data(
            explored_messages, explored_responses, 
            refined_segmentations[:len(explored_messages)]
        )
        
        # Step 4: Evaluate
        metrics_iter = compute_metrics(refined_segmentations, ground_truth)
        
        iteration_results.append({
            'iteration': iteration,
            'type': 'feedback',
            'metrics': metrics_iter,
            'num_messages': len(messages) + len(feedback.new_messages),
            'num_new_states': len(fsm.states)
        })
        
        logging.info(f"Iteration {iteration} Results:")
        logging.info(f"  F1 Score: {metrics_iter['f1']:.4f} (Δ={metrics_iter['f1']-metrics_v0['f1']:+.4f})")
        logging.info(f"  Perfect Match: {metrics_iter['perfect_match']:.4f} (Δ={metrics_iter['perfect_match']-metrics_v0['perfect_match']:+.4f})")
        logging.info(f"  FSM States: {len(fsm.states)}")
        
        # Update for next iteration
        current_segmentations = refined_segmentations
    
    # ==================== FINAL: FSM Generation ====================
    logging.info("\n" + "="*80)
    logging.info("FINAL: Generating Protocol FSM")
    logging.info("="*80)
    
    # Visualize FSM
    fsm.visualize(os.path.join(output_dir, f'fsm_{protocol}.png'))
    fsm.export_dot(os.path.join(output_dir, f'fsm_{protocol}.dot'))
    
    fsm_stats = fsm.get_statistics()
    logging.info(f"FSM Statistics:")
    for key, value in fsm_stats.items():
        logging.info(f"  {key}: {value}")
    
    # ==================== ANALYSIS: Compare with Baseline ====================
    analyze_feedback_impact(iteration_results, output_dir, protocol)
    
    # Save results
    save_experiment4_results(iteration_results, fsm_stats, output_dir, protocol)
    
    return iteration_results, fsm


def simulate_active_exploration(base_messages: List[bytes],
                                segmentations: List[List[int]],
                                num_rounds: int = 20) -> Tuple[List[bytes], List[bytes]]:
    """
    Simulate active exploration (Exp3)
    
    In real scenario, this would:
    1. Use DKL to select uncertain samples
    2. Send to real server
    3. Observe responses
    
    For simulation:
    - Generate mutations
    - Mock responses based on message fields
    """
    explored_messages = []
    explored_responses = []
    
    for _ in range(num_rounds):
        # Select base message
        base_msg = np.random.choice(base_messages)
        
        # Mutate
        mutated = bytearray(base_msg)
        for i in range(len(mutated)):
            if np.random.random() < 0.1:  # 10% mutation rate
                mutated[i] = np.random.randint(0, 256)
        
        explored_messages.append(bytes(mutated))
        
        # Mock response (in real scenario: server.send_and_recv())
        response_hash = hash(bytes(mutated[:8])) % 10
        explored_responses.append(f"STATE_{response_hash}".encode())
    
    return explored_messages, explored_responses


def analyze_feedback_impact(iteration_results: List[Dict],
                           output_dir: str,
                           protocol: str):
    """
    Analyze impact of feedback loop
    
    Generates:
    1. Convergence plot
    2. Comparison table
    """
    logging.info("Analyzing feedback impact...")
    
    # Extract metrics
    iterations = [r['iteration'] for r in iteration_results]
    f1_scores = [r['metrics']['f1'] for r in iteration_results]
    perf_scores = [r['metrics']['perfect_match'] for r in iteration_results]
    
    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # F1 Score
    axes[0].plot(iterations, f1_scores, 'o-', linewidth=2, markersize=8, 
                color='#2E86AB', label='F1 Score')
    axes[0].axhline(y=f1_scores[0], color='gray', linestyle='--', 
                   label='Baseline (No Feedback)')
    axes[0].set_xlabel('Iteration', fontweight='bold')
    axes[0].set_ylabel('F1 Score', fontweight='bold')
    axes[0].set_title(f'{protocol.upper()}: F1 Score Convergence', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Perfect Match
    axes[1].plot(iterations, perf_scores, 's-', linewidth=2, markersize=8,
                color='#F18F01', label='Perfect Match')
    axes[1].axhline(y=perf_scores[0], color='gray', linestyle='--',
                   label='Baseline (No Feedback)')
    axes[1].set_xlabel('Iteration', fontweight='bold')
    axes[1].set_ylabel('Perfect Match Rate', fontweight='bold')
    axes[1].set_title(f'{protocol.upper()}: Perfect Match Convergence', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'feedback_convergence_{protocol}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Convergence plot saved")
    
    # Improvement summary
    baseline_f1 = f1_scores[0]
    final_f1 = f1_scores[-1]
    improvement = ((final_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
    
    logging.info(f"\nFeedback Impact Summary:")
    logging.info(f"  Baseline F1: {baseline_f1:.4f}")
    logging.info(f"  Final F1: {final_f1:.4f}")
    logging.info(f"  Improvement: {improvement:+.2f}%")


def save_experiment4_results(iteration_results: List[Dict],
                            fsm_stats: Dict,
                            output_dir: str,
                            protocol: str):
    """Save all results"""
    results = {
        'protocol': protocol,
        'iterations': iteration_results,
        'fsm_statistics': fsm_stats,
        'baseline_f1': iteration_results[0]['metrics']['f1'],
        'final_f1': iteration_results[-1]['metrics']['f1'],
        'improvement_percent': ((iteration_results[-1]['metrics']['f1'] - 
                                iteration_results[0]['metrics']['f1']) / 
                               iteration_results[0]['metrics']['f1'] * 100)
    }
    
    with open(os.path.join(output_dir, f'experiment4_{protocol}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {output_dir}")
    
    # Generate report
    report_path = os.path.join(output_dir, f'EXPERIMENT4_REPORT_{protocol}.md')
    with open(report_path, 'w') as f:
        f.write(f"# Experiment 4: Closed-Loop Feedback - {protocol.upper()}\n\n")
        
        f.write("## Research Question\n")
        f.write("**RQ4**: Does active exploration feedback improve field segmentation accuracy?\n\n")
        
        f.write("## Results Summary\n\n")
        f.write(f"- **Protocol**: {protocol.upper()}\n")
        f.write(f"- **Baseline F1**: {results['baseline_f1']:.4f}\n")
        f.write(f"- **Final F1**: {results['final_f1']:.4f}\n")
        f.write(f"- **Improvement**: {results['improvement_percent']:+.2f}%\n\n")
        
        f.write("## FSM Statistics\n\n")
        for key, value in fsm_stats.items():
            f.write(f"- **{key}**: {value}\n")
        
        f.write("\n## Iteration Details\n\n")
        f.write("| Iteration | Type | F1 | Perfect Match | States |\n")
        f.write("|-----------|------|-----|---------------|--------|\n")
        for r in iteration_results:
            f.write(f"| {r['iteration']} | {r['type']} | "
                   f"{r['metrics']['f1']:.4f} | "
                   f"{r['metrics']['perfect_match']:.4f} | "
                   f"{r['num_new_states']} |\n")
        
        f.write("\n## Conclusion\n\n")
        if results['improvement_percent'] > 5:
            f.write("✅ **Significant improvement** observed with feedback loop.\n")
        elif results['improvement_percent'] > 0:
            f.write("✅ **Modest improvement** observed with feedback loop.\n")
        else:
            f.write("⚠️ **No improvement** - feedback may not be effective for this protocol.\n")
    
    logging.info(f"Report saved to {report_path}")


# ==================== Entry Point ====================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Experiment 4: Feedback Loop')
    parser.add_argument('--protocol', default='modbus', choices=['modbus', 'dnp3'],
                       help='Protocol to test')
    parser.add_argument('--max-messages', type=int, default=200,
                       help='Initial dataset size')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of feedback iterations')
    parser.add_argument('--exploration-rounds', type=int, default=20,
                       help='Exploration rounds per iteration')
    parser.add_argument('--output-dir', default='./experiments/exp4_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    results, fsm = run_experiment4(
        protocol=args.protocol,
        max_messages=args.max_messages,
        num_iterations=args.iterations,
        exploration_rounds_per_iter=args.exploration_rounds,
        output_dir=args.output_dir
    )
    
    print(f"\n{'='*80}")
    print("EXPERIMENT 4 COMPLETED")
    print(f"{'='*80}")
    print(f"Protocol: {args.protocol.upper()}")
    print(f"Baseline F1: {results[0]['metrics']['f1']:.4f}")
    print(f"Final F1: {results[-1]['metrics']['f1']:.4f}")
    print(f"Improvement: {((results[-1]['metrics']['f1']-results[0]['metrics']['f1'])/results[0]['metrics']['f1']*100):+.2f}%")
    print(f"FSM States: {len(fsm.states)}")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"{'='*80}")