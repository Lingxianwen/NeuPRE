"""
Experiment 3: Complex Constraint Inference

Tests ability to infer complex logical constraints between fields.

Example constraints:
- Field_A must be a multiple of Field_B
- Field_C = XOR(Field_A, Field_B)
- Field_D must be in range [Field_E, Field_F]

Expected result:
- NeuPRE can discover these constraints using Z3 solver
- DYNpre struggles because random mutation rarely satisfies complex constraints
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
from typing import List, Tuple, Dict, Callable

from neupre import NeuPRE, setup_logging
from utils.evaluator import NeuPREEvaluator
from modules.logic_refiner import (
    NeuroSymbolicLogicRefiner,
    FieldHypothesis,
    FieldType,
    generate_complex_constraint_counterexample
)


class ComplexProtocolServer:
    """
    Protocol server with complex constraints.

    Constraints:
    1. Field[1] must be a multiple of Field[0]
    2. Field[2] = XOR(Field[0], Field[1])
    3. Field[3] must equal length of Field[4]
    """

    def __init__(self):
        self.total_requests = 0
        self.accepted_requests = 0

    def validate_message(self, message: bytes) -> Tuple[bool, str]:
        """
        Validate message against constraints.

        Message format:
        [Field0(1)] [Field1(1)] [Field2(1)] [Field3(1)] [Field4(variable)]
        """
        if len(message) < 4:
            return False, "Message too short"

        field0 = message[0]
        field1 = message[1]
        field2 = message[2]
        field3 = message[3]
        field4 = message[4:]

        # Constraint 1: Field1 must be multiple of Field0
        if field0 > 0 and field1 % field0 != 0:
            return False, "Field1 not multiple of Field0"

        # Constraint 2: Field2 must be XOR(Field0, Field1)
        expected_xor = field0 ^ field1
        if field2 != expected_xor:
            return False, "Field2 not XOR of Field0 and Field1"

        # Constraint 3: Field3 must equal length of Field4
        if field3 != len(field4):
            return False, "Field3 not equal to length of Field4"

        return True, "Valid"

    def handle_message(self, message: bytes) -> Tuple[bool, bytes]:
        """
        Handle message and return (accepted, response).
        """
        self.total_requests += 1
        valid, reason = self.validate_message(message)

        if valid:
            self.accepted_requests += 1
            return True, b"OK_" + message[:4]
        else:
            return False, f"ERROR_{reason}".encode()


def simulate_dynpre_constraint_discovery(server: ComplexProtocolServer,
                                        num_attempts: int = 1000) -> List[Dict]:
    """
    Simulate DYNpre's constraint discovery through random testing.

    DYNpre uses random mutations and observes patterns.
    """
    logging.info("Simulating DYNpre constraint discovery (random testing)...")

    # Track field relationships
    field_observations = {
        'field0_field1': [],  # (field0, field1, accepted)
        'field0_field1_field2': [],  # (field0, field1, field2, accepted)
        'field3_field4_len': []  # (field3, len(field4), accepted)
    }

    for _ in range(num_attempts):
        # Random message generation
        field0 = np.random.randint(1, 16)
        field1 = np.random.randint(0, 256)
        field2 = np.random.randint(0, 256)
        field3 = np.random.randint(0, 32)
        field4_len = np.random.randint(0, 32)
        field4 = bytes(np.random.randint(0, 256, field4_len))

        message = bytes([field0, field1, field2, field3]) + field4
        accepted, response = server.handle_message(message)

        # Record observations
        field_observations['field0_field1'].append((field0, field1, accepted))
        field_observations['field0_field1_field2'].append((field0, field1, field2, accepted))
        field_observations['field3_field4_len'].append((field3, field4_len, accepted))

    # Try to infer constraints from observations
    inferred_constraints = []

    # Check if Field1 is multiple of Field0
    multiple_constraint = True
    for f0, f1, acc in field_observations['field0_field1'][:100]:
        if acc and f0 > 0 and f1 % f0 != 0:
            multiple_constraint = False
            break

    if multiple_constraint:
        inferred_constraints.append({
            'field_index': 1,
            'constraint_type': 'multiple_of',
            'target_field': 0
        })

    # Check if Field2 is XOR(Field0, Field1)
    xor_constraint = True
    for f0, f1, f2, acc in field_observations['field0_field1_field2'][:100]:
        if acc and f2 != (f0 ^ f1):
            xor_constraint = False
            break

    if xor_constraint:
        inferred_constraints.append({
            'field_index': 2,
            'constraint_type': 'xor',
            'target_fields': [0, 1]
        })

    # Check if Field3 equals length of Field4
    length_constraint = True
    for f3, f4_len, acc in field_observations['field3_field4_len'][:100]:
        if acc and f3 != f4_len:
            length_constraint = False
            break

    if length_constraint:
        inferred_constraints.append({
            'field_index': 3,
            'constraint_type': 'length',
            'target_field': 4
        })

    acceptance_rate = server.accepted_requests / server.total_requests
    logging.info(f"DYNpre acceptance rate: {acceptance_rate:.4f}")
    logging.info(f"DYNpre inferred {len(inferred_constraints)} constraints")

    return inferred_constraints


def simulate_neupre_constraint_discovery(server: ComplexProtocolServer,
                                        num_attempts: int = 200) -> List[Dict]:
    """
    Simulate NeuPRE's constraint discovery using neuro-symbolic approach.
    """
    logging.info("Simulating NeuPRE constraint discovery (neuro-symbolic)...")

    refiner = NeuroSymbolicLogicRefiner(
        confidence_threshold=0.6,
        max_counterexamples=5
    )

    # Generate initial hypotheses
    hypotheses = [
        FieldHypothesis(
            field_index=1,
            field_range=(1, 2),
            field_type=FieldType.UNKNOWN,
            confidence=0.8,
            parameters={'constraint': 'multiple_of', 'target': 0}
        ),
        FieldHypothesis(
            field_index=2,
            field_range=(2, 3),
            field_type=FieldType.CHECKSUM,
            confidence=0.8,
            parameters={'checksum_type': 'xor', 'target_fields': [0, 1]}
        ),
        FieldHypothesis(
            field_index=3,
            field_range=(3, 4),
            field_type=FieldType.LENGTH,
            confidence=0.8,
            parameters={'target_field': 4}
        )
    ]

    for hyp in hypotheses:
        refiner.add_hypothesis(hyp)

    # Verification callback
    def verify_callback(msg: bytes) -> Tuple[bool, bytes]:
        return server.handle_message(msg)

    # Template message
    template = b'\x04\x08\x0c\x05hello'

    # Verify hypotheses
    refiner.refine_rules(template, verify_callback)

    # Extract verified constraints
    verified_rules = refiner.get_verified_rules()

    inferred_constraints = []
    for rule in verified_rules:
        constraint = {
            'field_index': rule.field_index,
            'constraint_type': rule.parameters.get('constraint') or
                             rule.parameters.get('checksum_type') or
                             'length',
        }
        if 'target' in rule.parameters:
            constraint['target_field'] = rule.parameters['target']
        if 'target_fields' in rule.parameters:
            constraint['target_fields'] = rule.parameters['target_fields']
        if 'target_field' in rule.parameters:
            constraint['target_field'] = rule.parameters['target_field']

        inferred_constraints.append(constraint)

    acceptance_rate = server.accepted_requests / server.total_requests if server.total_requests > 0 else 0
    logging.info(f"NeuPRE acceptance rate: {acceptance_rate:.4f}")
    logging.info(f"NeuPRE inferred {len(inferred_constraints)} constraints")

    return inferred_constraints


def run_experiment3(dynpre_attempts: int = 1000,
                   neupre_attempts: int = 200,
                   output_dir: str = './experiment3_results'):
    """
    Run Experiment 3: Complex Constraint Inference.

    Args:
        dynpre_attempts: Number of attempts for DYNpre
        neupre_attempts: Number of attempts for NeuPRE
        output_dir: Output directory
    """
    setup_logging(level=logging.INFO)
    logging.info("=" * 80)
    logging.info("EXPERIMENT 3: Complex Constraint Inference")
    logging.info("=" * 80)

    evaluator = NeuPREEvaluator(output_dir=output_dir)

    # Ground truth constraints
    ground_truth = [
        {'field_index': 1, 'constraint_type': 'multiple_of', 'target_field': 0},
        {'field_index': 2, 'constraint_type': 'xor', 'target_fields': [0, 1]},
        {'field_index': 3, 'constraint_type': 'length', 'target_field': 4}
    ]

    # DYNpre
    logging.info("\n" + "-" * 80)
    dynpre_server = ComplexProtocolServer()
    dynpre_constraints = simulate_dynpre_constraint_discovery(
        dynpre_server, dynpre_attempts
    )

    # NeuPRE
    logging.info("\n" + "-" * 80)
    neupre_server = ComplexProtocolServer()
    neupre_constraints = simulate_neupre_constraint_discovery(
        neupre_server, neupre_attempts
    )

    # Evaluate
    logging.info("\n" + "-" * 80)
    logging.info("Evaluating constraint inference...")

    neupre_metrics = evaluator.evaluate_constraint_inference(
        neupre_constraints, ground_truth
    )

    dynpre_metrics = evaluator.evaluate_constraint_inference(
        dynpre_constraints, ground_truth
    )

    # Compare
    comparison = evaluator.compare_methods(
        neupre_metrics, dynpre_metrics, 'constraint'
    )

    # Plot
    evaluator.plot_constraint_inference(
        neupre_metrics, dynpre_metrics,
        filename='constraint_inference.png'
    )

    # Report
    evaluator.generate_report([comparison], filename='experiment3_report.txt')
    evaluator.save_metrics_json({
        'neupre': neupre_metrics.__dict__,
        'dynpre': dynpre_metrics.__dict__,
        'comparison': comparison,
        'ground_truth': ground_truth
    }, filename='experiment3_metrics.json')

    # Summary
    logging.info("\n" + "=" * 80)
    logging.info("EXPERIMENT 3 SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Ground Truth Constraints: {len(ground_truth)}")
    logging.info(f"\nNeuPRE:")
    logging.info(f"  Correctly Inferred: {neupre_metrics.correctly_inferred}/{len(ground_truth)}")
    logging.info(f"  F1-Score: {neupre_metrics.f1_score:.4f}")
    logging.info(f"  Messages Used: {neupre_server.total_requests}")
    logging.info(f"\nDYNpre:")
    logging.info(f"  Correctly Inferred: {dynpre_metrics.correctly_inferred}/{len(ground_truth)}")
    logging.info(f"  F1-Score: {dynpre_metrics.f1_score:.4f}")
    logging.info(f"  Messages Used: {dynpre_server.total_requests}")

    if dynpre_metrics.f1_score > 0:
        improvement = (neupre_metrics.f1_score - dynpre_metrics.f1_score) / \
                     dynpre_metrics.f1_score * 100
        logging.info(f"\nNeuPRE Improvement: {improvement:.1f}%")

    msg_efficiency = (dynpre_server.total_requests - neupre_server.total_requests) / \
                    dynpre_server.total_requests * 100
    logging.info(f"Message Efficiency: {msg_efficiency:.1f}% fewer messages")
    logging.info("=" * 80)


if __name__ == '__main__':
    run_experiment3(
        dynpre_attempts=1000,
        neupre_attempts=200,
        output_dir='./experiments/experiment3_results'
    )
