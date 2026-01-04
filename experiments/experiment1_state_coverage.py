"""
Experiment 1: State Coverage Efficiency

Compares how quickly NeuPRE vs DYNpre discovers unique protocol states.

Expected result:
- NeuPRE curve should be steeper (reaches target coverage with fewer packets)
- Demonstrates Bayesian active learning is more efficient than random mutation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
from typing import List, Tuple
import time

from neupre import NeuPRE, setup_logging
from utils.evaluator import NeuPREEvaluator
from modules.state_explorer import DeepKernelStateExplorer


class MockProtocolServer:
    """
    Mock protocol server with defined state machine for testing.
    """

    def __init__(self, num_states: int = 10):
        """
        Args:
            num_states: Number of unique states in the protocol
        """
        self.num_states = num_states
        self.state_responses = {
            i: f"STATE_{i}_RESPONSE".encode() + bytes([i] * 10)
            for i in range(num_states)
        }

    def handle_message(self, message: bytes) -> bytes:
        """
        Handle protocol message and return response.

        State is determined by first byte of message.
        """
        if not message:
            return b"ERROR_EMPTY"

        # Simple state determination: hash message to state ID
        state_id = sum(message) % self.num_states
        return self.state_responses[state_id]


def simulate_dynpre_exploration(server: MockProtocolServer,
                                base_messages: List[bytes],
                                num_iterations: int = 100,
                                mutation_rate: float = 0.1) -> Tuple[List[int], List[int]]:
    """
    Simulate DYNpre's random mutation exploration.

    Args:
        server: Mock protocol server
        base_messages: Seed messages
        num_iterations: Number of exploration iterations
        mutation_rate: Mutation probability per byte

    Returns:
        (messages_sent, unique_states) histories
    """
    logging.info("Simulating DYNpre exploration (random mutation)...")

    messages_sent = []
    unique_states = []
    discovered_states = set()
    total_messages = 0

    for iteration in range(num_iterations):
        # Random mutation
        base_msg = np.random.choice(base_messages)
        mutated = bytearray(base_msg)

        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] = np.random.randint(0, 256)

        # Send and observe
        response = server.handle_message(bytes(mutated))
        total_messages += 1

        # Track state
        state_hash = hash(response)
        discovered_states.add(state_hash)

        messages_sent.append(total_messages)
        unique_states.append(len(discovered_states))

        if iteration % 10 == 0:
            logging.debug(f"DYNpre iteration {iteration}: {len(discovered_states)} states")

    logging.info(f"DYNpre: {total_messages} messages → {len(discovered_states)} states")
    return messages_sent, unique_states


def simulate_neupre_exploration(server: MockProtocolServer,
                               base_messages: List[bytes],
                               num_iterations: int = 100) -> Tuple[List[int], List[int]]:
    """
    Simulate NeuPRE's active exploration.

    Args:
        server: Mock protocol server
        base_messages: Seed messages
        num_iterations: Number of exploration iterations

    Returns:
        (messages_sent, unique_states) histories
    """
    logging.info("Simulating NeuPRE exploration (active learning)...")

    # Initialize state explorer
    explorer = DeepKernelStateExplorer(
        embedding_dim=64,
        hidden_dim=128,
        feature_dim=32,
        kappa=2.0
    )

    def probe_callback(msg: bytes) -> bytes:
        return server.handle_message(msg)

    # Run active exploration
    stats = explorer.active_exploration(
        base_messages=base_messages,
        num_iterations=num_iterations,
        num_mutations=50,
        probe_callback=probe_callback
    )

    messages_sent = stats['iterations']
    unique_states = stats['unique_states']

    logging.info(f"NeuPRE: {len(messages_sent)} messages → {unique_states[-1]} states")
    return messages_sent, unique_states


def run_experiment1(num_states: int = 10,
                   num_iterations: int = 100,
                   num_runs: int = 5,
                   output_dir: str = './experiment1_results'):
    """
    Run Experiment 1: State Coverage Efficiency.

    Args:
        num_states: Number of states in mock protocol
        num_iterations: Exploration iterations per run
        num_runs: Number of runs to average over
        output_dir: Output directory
    """
    setup_logging(level=logging.INFO)
    logging.info("=" * 80)
    logging.info("EXPERIMENT 1: State Coverage Efficiency")
    logging.info("=" * 80)

    evaluator = NeuPREEvaluator(output_dir=output_dir)

    # Initialize mock server
    server = MockProtocolServer(num_states=num_states)

    # Generate seed messages
    base_messages = [
        bytes([i, i+1, i+2, i+3]) for i in range(10)
    ]

    # Run multiple times and average
    neupre_all_runs = []
    dynpre_all_runs = []

    for run in range(num_runs):
        logging.info(f"\n--- Run {run + 1}/{num_runs} ---")

        # NeuPRE exploration
        neupre_msgs, neupre_states = simulate_neupre_exploration(
            server, base_messages, num_iterations
        )
        neupre_all_runs.append((neupre_msgs, neupre_states))

        # DYNpre exploration
        dynpre_msgs, dynpre_states = simulate_dynpre_exploration(
            server, base_messages, num_iterations
        )
        dynpre_all_runs.append((dynpre_msgs, dynpre_states))

    # Average results
    neupre_msgs_avg = np.mean([msgs for msgs, _ in neupre_all_runs], axis=0).astype(int).tolist()
    neupre_states_avg = np.mean([states for _, states in neupre_all_runs], axis=0).tolist()

    dynpre_msgs_avg = np.mean([msgs for msgs, _ in dynpre_all_runs], axis=0).astype(int).tolist()
    dynpre_states_avg = np.mean([states for _, states in dynpre_all_runs], axis=0).tolist()

    # Evaluate
    target_coverage = int(num_states * 0.8)  # 80% of states

    neupre_metrics = evaluator.evaluate_state_coverage(
        neupre_msgs_avg, neupre_states_avg, target_coverage
    )

    dynpre_metrics = evaluator.evaluate_state_coverage(
        dynpre_msgs_avg, dynpre_states_avg, target_coverage
    )

    # Compare
    comparison = evaluator.compare_methods(
        neupre_metrics, dynpre_metrics, 'coverage'
    )

    # Plot
    evaluator.plot_state_coverage_curve(
        (neupre_msgs_avg, neupre_states_avg),
        (dynpre_msgs_avg, dynpre_states_avg),
        filename='state_coverage_efficiency.png'
    )

    # Generate report
    evaluator.generate_report([comparison], filename='experiment1_report.txt')
    evaluator.save_metrics_json({
        'neupre': neupre_metrics.__dict__,
        'dynpre': dynpre_metrics.__dict__,
        'comparison': comparison
    }, filename='experiment1_metrics.json')

    # Summary
    logging.info("\n" + "=" * 80)
    logging.info("EXPERIMENT 1 SUMMARY")
    logging.info("=" * 80)
    if neupre_metrics.messages_to_target_coverage and dynpre_metrics.messages_to_target_coverage:
        improvement = (dynpre_metrics.messages_to_target_coverage -
                      neupre_metrics.messages_to_target_coverage) / \
                      dynpre_metrics.messages_to_target_coverage * 100
        logging.info(f"NeuPRE reached {target_coverage} states in {neupre_metrics.messages_to_target_coverage} messages")
        logging.info(f"DYNpre reached {target_coverage} states in {dynpre_metrics.messages_to_target_coverage} messages")
        logging.info(f"NeuPRE improvement: {improvement:.1f}% fewer messages")
    logging.info("=" * 80)


if __name__ == '__main__':
    run_experiment1(
        num_states=15,
        num_iterations=100,
        num_runs=3,
        output_dir='./experiments/experiment1_results'
    )
