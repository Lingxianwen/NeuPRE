"""
NeuPRE: Neuro-Symbolic Protocol Reverse Engineering
Main pipeline integrating all three modules
"""

import logging
import time
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
import pickle
import os
from collections import defaultdict

from modules.format_learner import InformationBottleneckFormatLearner
from modules.state_explorer import DeepKernelStateExplorer
from modules.logic_refiner import NeuroSymbolicLogicRefiner, FieldHypothesis, FieldType


class NeuPRE:
    """
    Main NeuPRE pipeline.

    Integrates three modules:
    1. Information Bottleneck Format Learner - Automatic field segmentation
    2. Deep Kernel Learning State Explorer - Active state discovery
    3. Neuro-Symbolic Logic Refiner - Constraint inference and verification
    """

    def __init__(self,
                 # Format Learner params
                 ib_d_model: int = 128,
                 ib_nhead: int = 4,
                 ib_layers: int = 3,
                 ib_beta: float = 0.1,
                 # State Explorer params
                 dkl_embedding_dim: int = 128,
                 dkl_hidden_dim: int = 256,
                 dkl_feature_dim: int = 64,
                 dkl_kappa: float = 2.0,
                 # Logic Refiner params
                 confidence_threshold: float = 0.7,
                 max_counterexamples: int = 10,
                 # General params
                 device: str = 'cuda',
                 output_dir: str = './neupre_output'):
        """
        Initialize NeuPRE pipeline.

        Args:
            ib_d_model: Transformer dimension for format learner
            ib_nhead: Number of attention heads
            ib_layers: Number of transformer layers
            ib_beta: IB trade-off parameter
            dkl_embedding_dim: Embedding dimension for state explorer
            dkl_hidden_dim: Hidden dimension for state explorer
            dkl_feature_dim: Feature dimension for state explorer
            dkl_kappa: UCB exploration parameter
            confidence_threshold: Minimum confidence for hypothesis
            max_counterexamples: Maximum counter-examples to test
            device: Device to use (cuda/cpu)
            output_dir: Output directory for results
        """
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize modules
        logging.info("=" * 80)
        logging.info("Initializing NeuPRE Pipeline")
        logging.info("=" * 80)

        self.format_learner = InformationBottleneckFormatLearner(
            d_model=ib_d_model,
            nhead=ib_nhead,
            num_layers=ib_layers,
            beta=ib_beta,
            device=device
        )

        self.state_explorer = DeepKernelStateExplorer(
            embedding_dim=dkl_embedding_dim,
            hidden_dim=dkl_hidden_dim,
            feature_dim=dkl_feature_dim,
            kappa=dkl_kappa,
            device=device
        )

        self.logic_refiner = NeuroSymbolicLogicRefiner(
            confidence_threshold=confidence_threshold,
            max_counterexamples=max_counterexamples
        )

        # Data storage
        self.messages = []
        self.responses = []
        self.segmentations = []
        self.field_hypotheses = []
        self.verified_rules = []

        logging.info("NeuPRE initialization complete")
        logging.info("=" * 80)

    def phase1_format_learning(self, messages: List[bytes],
                               responses: Optional[List[bytes]] = None,
                               epochs: int = 50,
                               batch_size: int = 32) -> List[List[Tuple[int, int]]]:
        """
        Phase 1: Format Learning using Information Bottleneck.

        Args:
            messages: List of protocol messages
            responses: Optional list of server responses
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            List of segmentations (one per message)
        """
        logging.info("=" * 80)
        logging.info("PHASE 1: Format Learning (Information Bottleneck)")
        logging.info("=" * 80)

        # Train format learner
        self.format_learner.train(
            messages=messages,
            responses=responses,
            epochs=epochs,
            batch_size=batch_size
        )

        # Extract segmentations for all messages
        segmentations = []
        for i, msg in enumerate(messages):
            segments = self.format_learner.segment_message(msg, threshold=0.5)
            segmentations.append(segments)

            if i < 3:  # Log first few
                logging.info(f"Message {i} segmentation: {segments}")

        self.messages = messages
        self.responses = responses if responses else []
        self.segmentations = segmentations

        logging.info(f"Format learning complete. Segmented {len(messages)} messages")
        logging.info("=" * 80)

        return segmentations

    def phase2_state_exploration(self,
                                base_messages: List[bytes],
                                num_iterations: int = 100,
                                num_mutations: int = 50,
                                probe_callback: Optional[Callable[[bytes], bytes]] = None,
                                train_epochs: int = 10) -> Dict:
        """
        Phase 2: Active State Exploration using Deep Kernel Learning.

        Args:
            base_messages: Seed messages for exploration
            num_iterations: Number of exploration iterations
            num_mutations: Number of mutations per iteration
            probe_callback: Function to send message and get response
            train_epochs: Training epochs for DKL

        Returns:
            Exploration statistics
        """
        logging.info("=" * 80)
        logging.info("PHASE 2: State Exploration (Deep Kernel Learning)")
        logging.info("=" * 80)

        # Perform active exploration
        stats = self.state_explorer.active_exploration(
            base_messages=base_messages,
            num_iterations=num_iterations,
            num_mutations=num_mutations,
            probe_callback=probe_callback
        )

        # Final training on all observed data
        if len(self.state_explorer.observed_messages) > 10:
            logging.info("Final DKL training on all observed data...")
            self.state_explorer.train(epochs=train_epochs)

        logging.info(f"State exploration complete. "
                   f"Discovered {stats['unique_states'][-1]} unique states")
        logging.info("=" * 80)

        return stats

    def phase3_logic_refinement(self,
                                message_template: bytes,
                                probe_callback: Callable[[bytes], Tuple[bool, bytes]]) -> List[FieldHypothesis]:
        """
        Phase 3: Logic Refinement using Neuro-Symbolic reasoning.

        Args:
            message_template: Template message for structure
            probe_callback: Function to test messages (returns accepted, response)

        Returns:
            List of verified field hypotheses
        """
        logging.info("=" * 80)
        logging.info("PHASE 3: Logic Refinement (Neuro-Symbolic)")
        logging.info("=" * 80)

        # Generate initial hypotheses from format learning
        if self.segmentations and self.messages:
            hypotheses = self.logic_refiner.infer_field_types(
                self.messages, self.segmentations
            )

            for hyp in hypotheses:
                self.logic_refiner.add_hypothesis(hyp)

        # Refine hypotheses using SMT solver
        self.logic_refiner.refine_rules(
            message_template=message_template,
            probe_callback=probe_callback
        )

        # Get verified rules
        self.verified_rules = self.logic_refiner.get_verified_rules()

        summary = self.logic_refiner.get_rule_summary()
        logging.info(f"Logic refinement complete:")
        logging.info(f"  Total hypotheses: {summary['total_hypotheses']}")
        logging.info(f"  Verified: {summary['verified']}")
        logging.info(f"  Rejected: {summary['rejected']}")
        logging.info(f"  Verification rate: {summary['verification_rate']:.2%}")
        logging.info("=" * 80)

        return self.verified_rules

    def run_full_pipeline(self,
                         initial_messages: List[bytes],
                         initial_responses: Optional[List[bytes]] = None,
                         probe_callback: Optional[Callable[[bytes], bytes]] = None,
                         verify_callback: Optional[Callable[[bytes], Tuple[bool, bytes]]] = None,
                         # Phase 1 params
                         format_epochs: int = 50,
                         format_batch_size: int = 32,
                         # Phase 2 params
                         exploration_iterations: int = 100,
                         exploration_mutations: int = 50,
                         # Phase 3 params
                         enable_logic_refinement: bool = True) -> Dict:
        """
        Run complete NeuPRE pipeline.

        Args:
            initial_messages: Initial protocol messages
            initial_responses: Initial responses (optional)
            probe_callback: Function for exploration (msg -> response)
            verify_callback: Function for verification (msg -> (accepted, response))
            format_epochs: Epochs for format learning
            format_batch_size: Batch size for format learning
            exploration_iterations: Iterations for state exploration
            exploration_mutations: Mutations per iteration
            enable_logic_refinement: Enable phase 3

        Returns:
            Dictionary with all results
        """
        start_time = time.time()

        results = {
            'phase1': {},
            'phase2': {},
            'phase3': {},
            'total_time': 0
        }

        # Phase 1: Format Learning
        phase1_start = time.time()
        segmentations = self.phase1_format_learning(
            messages=initial_messages,
            responses=initial_responses,
            epochs=format_epochs,
            batch_size=format_batch_size
        )
        results['phase1']['segmentations'] = segmentations
        results['phase1']['time'] = time.time() - phase1_start

        # Phase 2: State Exploration
        if probe_callback is not None:
            phase2_start = time.time()
            stats = self.phase2_state_exploration(
                base_messages=initial_messages,
                num_iterations=exploration_iterations,
                num_mutations=exploration_mutations,
                probe_callback=probe_callback
            )
            results['phase2']['stats'] = stats
            results['phase2']['unique_states'] = self.state_explorer.get_state_coverage()
            results['phase2']['time'] = time.time() - phase2_start
        else:
            logging.warning("Skipping Phase 2: No probe callback provided")

        # Phase 3: Logic Refinement
        if enable_logic_refinement and verify_callback is not None:
            phase3_start = time.time()
            message_template = initial_messages[0] if initial_messages else b'\x00' * 64
            verified_rules = self.phase3_logic_refinement(
                message_template=message_template,
                probe_callback=verify_callback
            )
            results['phase3']['verified_rules'] = [
                {
                    'field_index': r.field_index,
                    'field_range': r.field_range,
                    'field_type': r.field_type.value,
                    'confidence': r.confidence,
                    'parameters': r.parameters
                }
                for r in verified_rules
            ]
            results['phase3']['time'] = time.time() - phase3_start
        else:
            logging.warning("Skipping Phase 3: Disabled or no verify callback")

        results['total_time'] = time.time() - start_time

        # Save results
        self.save_results(results)

        logging.info("=" * 80)
        logging.info("NeuPRE Pipeline Complete")
        logging.info(f"Total time: {results['total_time']:.2f}s")
        logging.info("=" * 80)

        return results

    def save_results(self, results: Dict, filename: str = 'neupre_results.pkl'):
        """Save results to file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        logging.info(f"Results saved to {filepath}")

    def load_results(self, filename: str = 'neupre_results.pkl') -> Dict:
        """Load results from file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        logging.info(f"Results loaded from {filepath}")
        return results

    def save_models(self):
        """Save all trained models"""
        self.format_learner.save_model(
            os.path.join(self.output_dir, 'format_learner.pt')
        )
        self.state_explorer.save_model(
            os.path.join(self.output_dir, 'state_explorer.pt')
        )
        logging.info("Models saved")

    def load_models(self):
        """Load all trained models"""
        self.format_learner.load_model(
            os.path.join(self.output_dir, 'format_learner.pt')
        )
        self.state_explorer.load_model(
            os.path.join(self.output_dir, 'state_explorer.pt')
        )
        logging.info("Models loaded")

    def export_protocol_specification(self, filename: str = 'protocol_spec.txt'):
        """
        Export discovered protocol specification in human-readable format.

        Args:
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NeuPRE Protocol Specification\n")
            f.write("=" * 80 + "\n\n")

            # Format information
            f.write("## Message Format\n\n")
            if self.segmentations:
                f.write(f"Example segmentation (first message):\n")
                for i, (start, end) in enumerate(self.segmentations[0]):
                    f.write(f"  Field {i}: bytes [{start}:{end}] (length={end-start})\n")
                f.write("\n")

            # Field types
            f.write("## Field Types\n\n")
            if self.verified_rules:
                for rule in self.verified_rules:
                    f.write(f"Field {rule.field_index} "
                          f"[{rule.field_range[0]}:{rule.field_range[1]}]: "
                          f"{rule.field_type.value} "
                          f"(confidence={rule.confidence:.2f})\n")
                    if rule.parameters:
                        f.write(f"  Parameters: {rule.parameters}\n")
                    f.write("\n")
            else:
                f.write("No verified field types\n\n")

            # State information
            f.write("## State Space\n\n")
            f.write(f"Unique states discovered: {self.state_explorer.get_state_coverage()}\n")
            f.write(f"Total messages observed: {len(self.state_explorer.observed_messages)}\n\n")

            # Statistics
            f.write("## Statistics\n\n")
            summary = self.logic_refiner.get_rule_summary()
            f.write(f"Total hypotheses: {summary['total_hypotheses']}\n")
            f.write(f"Verified rules: {summary['verified']}\n")
            f.write(f"Rejected rules: {summary['rejected']}\n")
            f.write(f"Verification rate: {summary['verification_rate']:.2%}\n")

            f.write("\n" + "=" * 80 + "\n")

        logging.info(f"Protocol specification exported to {filepath}")


def setup_logging(level=logging.INFO, logfile: Optional[str] = None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]

    if logfile:
        handlers.append(logging.FileHandler(logfile))

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
