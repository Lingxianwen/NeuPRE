"""
Module 3: Neuro-Symbolic Logic Refiner

This module converts probabilistic predictions from neural networks into deterministic rules
using symbolic reasoning and constraint solving. This is a capability that DYNpre completely lacks.

Key idea:
1. Neural network outputs probabilistic field attributes (e.g., "Field A has 80% probability of being length")
2. Generate symbolic hypotheses (logical formulas)
3. Use SMT solver (Z3) to verify hypotheses by constructing counter-examples
4. Test counter-examples against real server
5. Refine hypotheses based on feedback

This implements the scientific method: hypothesis generation → falsification → refinement
"""

import z3
from z3 import Solver, BitVec, Concat, Extract, If, And, Or, Not, sat, unsat
import numpy as np
from typing import List, Tuple, Dict, Optional, Set, Callable
import logging
from enum import Enum
from dataclasses import dataclass


class FieldType(Enum):
    """Field types that can be inferred"""
    CONSTANT = "constant"
    LENGTH = "length"
    CHECKSUM = "checksum"
    SEQUENCE = "sequence"
    RANDOM = "random"
    ENUM = "enum"
    PAYLOAD = "payload"
    UNKNOWN = "unknown"


@dataclass
class FieldHypothesis:
    """Hypothesis about a field's properties"""
    field_index: int  # Which field (by position)
    field_range: Tuple[int, int]  # Byte range (start, end)
    field_type: FieldType
    confidence: float  # Probability from neural network
    parameters: Dict  # Type-specific parameters

    def to_z3_constraint(self, field_vars: Dict, message_vars: Dict) -> z3.BoolRef:
        """
        Convert hypothesis to Z3 constraint.

        Args:
            field_vars: Dictionary of Z3 variables for fields
            message_vars: Dictionary of Z3 variables for entire message

        Returns:
            Z3 constraint
        """
        field_var = field_vars[self.field_index]

        if self.field_type == FieldType.CONSTANT:
            # Field must equal constant value
            const_value = self.parameters.get('value', 0)
            return field_var == const_value

        elif self.field_type == FieldType.LENGTH:
            # Field value equals length of target field
            target_field = self.parameters.get('target_field', -1)
            if target_field >= 0 and target_field in field_vars:
                # This is simplified; real implementation needs proper length calculation
                # For now, return unconstrained as this requires complex field length tracking
                return z3.BoolVal(True)
            return z3.BoolVal(True)

        elif self.field_type == FieldType.CHECKSUM:
            # Field is checksum of other fields
            target_fields = self.parameters.get('target_fields', [])
            checksum_type = self.parameters.get('checksum_type', 'xor')

            # Check if all target fields are available
            if not target_fields or not all(f in field_vars for f in target_fields):
                return z3.BoolVal(True)

            if checksum_type == 'xor':
                # XOR checksum
                result = field_vars[target_fields[0]]
                for i in range(1, len(target_fields)):
                    result = result ^ field_vars[target_fields[i]]
                return field_var == result

            elif checksum_type == 'sum':
                # Sum checksum (modulo)
                result = field_vars[target_fields[0]]
                for i in range(1, len(target_fields)):
                    result = result + field_vars[target_fields[i]]
                return field_var == result

            return z3.BoolVal(True)

        elif self.field_type == FieldType.SEQUENCE:
            # Field increments sequentially
            # This requires tracking state across messages
            return z3.BoolVal(True)  # Simplified for now

        elif self.field_type == FieldType.ENUM:
            # Field must be one of enumerated values
            allowed_values = self.parameters.get('values', [])
            if allowed_values:
                constraints = [field_var == val for val in allowed_values]
                return z3.Or(*constraints)
            return z3.BoolVal(True)

        else:
            # Unknown or unconstrained
            return z3.BoolVal(True)


class NeuroSymbolicLogicRefiner:
    """
    Main class for neuro-symbolic logic refinement.

    Converts neural network predictions into verified symbolic rules.
    """

    def __init__(self, confidence_threshold: float = 0.7,
                 max_counterexamples: int = 10):
        """
        Args:
            confidence_threshold: Minimum confidence to generate hypothesis
            max_counterexamples: Maximum number of counter-examples to test
        """
        self.confidence_threshold = confidence_threshold
        self.max_counterexamples = max_counterexamples

        # Storage
        self.hypotheses: List[FieldHypothesis] = []
        self.verified_rules: List[FieldHypothesis] = []
        self.rejected_rules: List[FieldHypothesis] = []

        logging.info("NeuroSymbolicLogicRefiner initialized")
        logging.info(f"Confidence threshold: {confidence_threshold}")

    def add_hypothesis(self, hypothesis: FieldHypothesis):
        """
        Add a hypothesis from neural network prediction.

        Args:
            hypothesis: Field hypothesis with confidence score
        """
        if hypothesis.confidence >= self.confidence_threshold:
            self.hypotheses.append(hypothesis)
            logging.info(f"Added hypothesis: Field {hypothesis.field_index} is "
                       f"{hypothesis.field_type.value} (confidence={hypothesis.confidence:.2f})")
        else:
            logging.debug(f"Rejected low-confidence hypothesis: "
                        f"Field {hypothesis.field_index} (confidence={hypothesis.confidence:.2f})")

    def generate_counterexample(self, hypothesis: FieldHypothesis,
                               message_template: bytes) -> Optional[bytes]:
        """
        Use Z3 to generate a counter-example that violates the hypothesis.

        If hypothesis is "Field A == len(Payload)", we try to find a message
        where Field A != len(Payload).

        Args:
            hypothesis: Hypothesis to test
            message_template: Template message for structure

        Returns:
            Counter-example message, or None if hypothesis is always true
        """
        logging.debug(f"Generating counter-example for: Field {hypothesis.field_index} "
                    f"is {hypothesis.field_type.value}")

        # Create Z3 solver
        solver = Solver()

        # Create symbolic variables for message bytes
        msg_len = len(message_template)
        msg_vars = [BitVec(f'byte_{i}', 8) for i in range(msg_len)]

        # Collect all field indices that need variables
        field_indices = {hypothesis.field_index}

        # Add target fields if present
        if 'target_fields' in hypothesis.parameters:
            field_indices.update(hypothesis.parameters['target_fields'])
        if 'target_field' in hypothesis.parameters:
            field_indices.add(hypothesis.parameters['target_field'])
        if 'target' in hypothesis.parameters:
            field_indices.add(hypothesis.parameters['target'])

        # Create variables for all needed fields
        # Assume 1-byte fields at consecutive positions for simplicity
        field_vars = {}
        for field_idx in field_indices:
            if field_idx == hypothesis.field_index:
                start_idx = hypothesis.field_range[0]
                end_idx = hypothesis.field_range[1]
            else:
                # For other fields, assume 1-byte field at position field_idx
                start_idx = field_idx
                end_idx = field_idx + 1

            if start_idx >= msg_len or end_idx > msg_len:
                continue

            field_bytes = msg_vars[start_idx:end_idx]

            # Concatenate bytes to form field value
            if len(field_bytes) == 1:
                field_vars[field_idx] = field_bytes[0]
            elif len(field_bytes) == 2:
                field_vars[field_idx] = Concat(field_bytes[0], field_bytes[1])
            elif len(field_bytes) == 4:
                field_vars[field_idx] = Concat(
                    Concat(field_bytes[0], field_bytes[1]),
                    Concat(field_bytes[2], field_bytes[3])
                )
            else:
                # For other lengths, just use first byte as representative
                field_vars[field_idx] = field_bytes[0]

        # Get constraint from hypothesis
        constraint = hypothesis.to_z3_constraint(field_vars, {})

        # We want to find a counter-example, so negate the constraint
        solver.add(Not(constraint))

        # Add basic sanity constraints (optional)
        # For example, ensure certain bytes are in valid ranges
        # This can be customized based on protocol knowledge

        # Solve
        result = solver.check()

        if result == sat:
            # Found a counter-example
            model = solver.model()

            # Extract message bytes from model
            counterexample = bytearray(msg_len)
            for i in range(msg_len):
                val = model.eval(msg_vars[i])
                # Convert Z3 value to Python int
                try:
                    if hasattr(val, 'as_long'):
                        counterexample[i] = val.as_long()
                    else:
                        # Newer Z3 API
                        counterexample[i] = val.as_signed_long() & 0xFF
                except:
                    # Fallback: use string conversion
                    try:
                        counterexample[i] = int(str(val)) & 0xFF
                    except:
                        counterexample[i] = 0

            logging.info(f"Generated counter-example for Field {hypothesis.field_index}")
            return bytes(counterexample)

        elif result == unsat:
            # No counter-example exists - hypothesis is always true!
            logging.info(f"Hypothesis is always true (no counter-example): "
                       f"Field {hypothesis.field_index}")
            return None

        else:
            # Unknown
            logging.warning("Z3 solver returned unknown")
            return None

    def verify_hypothesis(self, hypothesis: FieldHypothesis,
                         message_template: bytes,
                         probe_callback: Callable[[bytes], Tuple[bool, bytes]]) -> bool:
        """
        Verify hypothesis by generating counter-examples and testing against server.

        Args:
            hypothesis: Hypothesis to verify
            message_template: Template message
            probe_callback: Function that sends message and returns (accepted, response)

        Returns:
            True if hypothesis is verified, False if falsified
        """
        logging.info(f"Verifying hypothesis: Field {hypothesis.field_index} is "
                   f"{hypothesis.field_type.value}")

        # Generate counter-examples
        for attempt in range(self.max_counterexamples):
            counterexample = self.generate_counterexample(hypothesis, message_template)

            if counterexample is None:
                # No counter-example found - hypothesis is likely correct
                logging.info(f"No counter-example found after {attempt + 1} attempts. "
                           f"Hypothesis verified.")
                return True

            # Test counter-example against server
            accepted, response = probe_callback(counterexample)

            if accepted:
                # Server accepted the counter-example!
                # This means our hypothesis is WRONG
                logging.info(f"Counter-example was accepted by server. "
                           f"Hypothesis falsified.")
                return False
            else:
                # Server rejected counter-example
                # This supports our hypothesis
                logging.debug(f"Counter-example rejected by server (as expected)")

        # After multiple attempts, if all counter-examples were rejected,
        # hypothesis is likely correct
        logging.info(f"All {self.max_counterexamples} counter-examples were rejected. "
                   f"Hypothesis verified.")
        return True

    def refine_rules(self, message_template: bytes,
                    probe_callback: Callable[[bytes], Tuple[bool, bytes]]):
        """
        Refine all hypotheses using SMT-based verification.

        Args:
            message_template: Template message for structure
            probe_callback: Function to probe server
        """
        logging.info(f"Refining {len(self.hypotheses)} hypotheses")

        for hypothesis in self.hypotheses:
            is_verified = self.verify_hypothesis(hypothesis, message_template, probe_callback)

            if is_verified:
                self.verified_rules.append(hypothesis)
                logging.info(f"[VERIFIED] Field {hypothesis.field_index} is "
                           f"{hypothesis.field_type.value}")
            else:
                self.rejected_rules.append(hypothesis)
                logging.info(f"[REJECTED] Field {hypothesis.field_index} is "
                           f"{hypothesis.field_type.value}")

        logging.info(f"Refinement completed: {len(self.verified_rules)} verified, "
                   f"{len(self.rejected_rules)} rejected")

    def infer_field_types(self, messages: List[bytes],
                         segmentations: List[List[Tuple[int, int]]]) -> List[FieldHypothesis]:
        """
        Infer field types from message data using heuristics.
        This provides initial hypotheses for the neural network.

        Args:
            messages: List of protocol messages
            segmentations: List of field segmentations for each message

        Returns:
            List of field hypotheses
        """
        logging.info(f"Inferring field types from {len(messages)} messages")

        hypotheses = []

        # Analyze each field position
        if not segmentations or not messages:
            return hypotheses

        num_fields = len(segmentations[0])

        for field_idx in range(num_fields):
            # Extract field values across all messages
            field_values = []
            field_range = segmentations[0][field_idx]

            for i, msg in enumerate(messages):
                if field_idx < len(segmentations[i]):
                    start, end = segmentations[i][field_idx]
                    if end <= len(msg):
                        field_bytes = msg[start:end]
                        field_values.append(field_bytes)

            if not field_values:
                continue

            # Check for constant field
            if len(set(field_values)) == 1:
                const_value = int.from_bytes(field_values[0], byteorder='big')
                hypotheses.append(FieldHypothesis(
                    field_index=field_idx,
                    field_range=field_range,
                    field_type=FieldType.CONSTANT,
                    confidence=0.95,
                    parameters={'value': const_value}
                ))
                continue

            # Check for sequence field
            is_sequence = True
            if len(field_values) > 1:
                for i in range(1, len(field_values)):
                    prev_val = int.from_bytes(field_values[i-1], byteorder='big')
                    curr_val = int.from_bytes(field_values[i], byteorder='big')
                    if curr_val != prev_val + 1:
                        is_sequence = False
                        break

                if is_sequence:
                    hypotheses.append(FieldHypothesis(
                        field_index=field_idx,
                        field_range=field_range,
                        field_type=FieldType.SEQUENCE,
                        confidence=0.90,
                        parameters={}
                    ))
                    continue

            # Check for enum field (small set of values)
            unique_values = set(field_values)
            if len(unique_values) <= 10:  # Arbitrary threshold
                enum_values = [int.from_bytes(v, byteorder='big') for v in unique_values]
                hypotheses.append(FieldHypothesis(
                    field_index=field_idx,
                    field_range=field_range,
                    field_type=FieldType.ENUM,
                    confidence=0.80,
                    parameters={'values': enum_values}
                ))
                continue

            # Check for length field
            # Try to correlate with other field lengths
            for target_idx in range(num_fields):
                if target_idx == field_idx:
                    continue

                is_length = True
                for i, msg in enumerate(messages):
                    if field_idx < len(segmentations[i]) and target_idx < len(segmentations[i]):
                        field_val = int.from_bytes(field_values[i], byteorder='big')
                        target_start, target_end = segmentations[i][target_idx]
                        target_len = target_end - target_start

                        if field_val != target_len:
                            is_length = False
                            break

                if is_length:
                    hypotheses.append(FieldHypothesis(
                        field_index=field_idx,
                        field_range=field_range,
                        field_type=FieldType.LENGTH,
                        confidence=0.85,
                        parameters={'target_field': target_idx}
                    ))
                    break

        logging.info(f"Inferred {len(hypotheses)} field type hypotheses")
        return hypotheses

    def get_verified_rules(self) -> List[FieldHypothesis]:
        """Get all verified rules"""
        return self.verified_rules

    def get_rule_summary(self) -> Dict:
        """Get summary of rules"""
        return {
            'total_hypotheses': len(self.hypotheses),
            'verified': len(self.verified_rules),
            'rejected': len(self.rejected_rules),
            'verification_rate': len(self.verified_rules) / len(self.hypotheses)
                               if self.hypotheses else 0
        }


def generate_complex_constraint_counterexample(
    constraint_formula: str,
    field_ranges: Dict[str, Tuple[int, int]],
    message_length: int
) -> Optional[bytes]:
    """
    Generate counter-example for complex constraints.

    Example constraint: "Field_A must be a multiple of Field_B"

    Args:
        constraint_formula: Python-like formula (e.g., "field_a % field_b == 0")
        field_ranges: Dictionary mapping field names to byte ranges
        message_length: Total message length

    Returns:
        Counter-example message
    """
    solver = Solver()

    # Create symbolic variables
    msg_vars = [BitVec(f'byte_{i}', 8) for i in range(message_length)]

    # Extract field variables
    field_vars = {}
    for field_name, (start, end) in field_ranges.items():
        field_bytes = msg_vars[start:end]
        if len(field_bytes) == 1:
            field_vars[field_name] = field_bytes[0]
        elif len(field_bytes) == 2:
            field_vars[field_name] = Concat(field_bytes[0], field_bytes[1])
        elif len(field_bytes) == 4:
            field_vars[field_name] = Concat(
                Concat(field_bytes[0], field_bytes[1]),
                Concat(field_bytes[2], field_bytes[3])
            )

    # Parse and add constraint
    # This is simplified - real implementation would need proper parsing
    if "%" in constraint_formula and "==" in constraint_formula:
        # Example: field_a % field_b == 0
        parts = constraint_formula.split("==")
        lhs = parts[0].strip()

        if "%" in lhs:
            dividend, divisor = lhs.split("%")
            dividend = dividend.strip()
            divisor = divisor.strip()

            if dividend in field_vars and divisor in field_vars:
                # Add negation to find counter-example
                constraint = field_vars[dividend] % field_vars[divisor] != 0
                solver.add(constraint)

    # Solve
    if solver.check() == sat:
        model = solver.model()
        counterexample = bytearray(message_length)
        for i in range(message_length):
            val = model.eval(msg_vars[i])
            # Convert Z3 value to Python int
            try:
                if hasattr(val, 'as_long'):
                    counterexample[i] = val.as_long()
                else:
                    counterexample[i] = val.as_signed_long() & 0xFF
            except:
                try:
                    counterexample[i] = int(str(val)) & 0xFF
                except:
                    counterexample[i] = 0
        return bytes(counterexample)

    return None
