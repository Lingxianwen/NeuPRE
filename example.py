"""
Example: Running NeuPRE on a simple custom protocol

This example demonstrates how to use NeuPRE to reverse engineer
a simple custom protocol.
"""

import sys
import logging
from typing import Tuple

from neupre import NeuPRE, setup_logging


class SimpleProtocolServer:
    """
    Simple custom protocol for demonstration.

    Message format:
    [Header(2)] [Command(1)] [Length(1)] [Payload(variable)] [Checksum(1)]

    Commands:
    0x01: ECHO - returns payload
    0x02: UPPER - returns uppercase payload
    0x03: REVERSE - returns reversed payload
    """

    HEADER = b'\xAA\xBB'

    def __init__(self):
        self.message_count = 0

    def calculate_checksum(self, data: bytes) -> int:
        """Simple XOR checksum"""
        checksum = 0
        for b in data:
            checksum ^= b
        return checksum

    def handle_message(self, message: bytes) -> Tuple[bool, bytes]:
        """
        Process protocol message.

        Returns:
            (accepted, response)
        """
        self.message_count += 1

        # Validate minimum length
        if len(message) < 5:
            return False, b'ERROR: Message too short'

        # Validate header
        if message[:2] != self.HEADER:
            return False, b'ERROR: Invalid header'

        # Parse fields
        command = message[2]
        length = message[3]
        payload = message[4:4+length]
        checksum = message[4+length] if len(message) > 4+length else 0

        # Validate length
        if len(payload) != length:
            return False, b'ERROR: Length mismatch'

        # Validate checksum
        expected_checksum = self.calculate_checksum(message[:4+length])
        if checksum != expected_checksum:
            return False, b'ERROR: Checksum failed'

        # Process command
        if command == 0x01:  # ECHO
            response = self.HEADER + b'\x81' + payload
        elif command == 0x02:  # UPPER
            response = self.HEADER + b'\x82' + payload.upper()
        elif command == 0x03:  # REVERSE
            response = self.HEADER + b'\x83' + payload[::-1]
        else:
            return False, b'ERROR: Unknown command'

        return True, response


def create_valid_message(command: int, payload: bytes) -> bytes:
    """Helper to create valid protocol messages"""
    header = b'\xAA\xBB'
    cmd = bytes([command])
    length = bytes([len(payload)])
    message_no_checksum = header + cmd + length + payload

    checksum = 0
    for b in message_no_checksum:
        checksum ^= b

    return message_no_checksum + bytes([checksum])


def main():
    """Run NeuPRE on simple protocol example"""

    setup_logging(level=logging.INFO, logfile='example.log')

    logging.info("=" * 80)
    logging.info("NeuPRE Example: Simple Custom Protocol")
    logging.info("=" * 80)

    # Initialize server
    server = SimpleProtocolServer()

    # Create seed messages
    seed_messages = [
        create_valid_message(0x01, b'hello'),
        create_valid_message(0x01, b'world'),
        create_valid_message(0x02, b'test'),
        create_valid_message(0x03, b'reverse'),
    ]

    # Collect initial responses
    initial_responses = []
    for msg in seed_messages:
        accepted, response = server.handle_message(msg)
        if accepted:
            initial_responses.append(response)
        else:
            initial_responses.append(b'')
            logging.warning(f"Seed message rejected: {response}")

    # Define callbacks
    def probe_callback(msg: bytes) -> bytes:
        """Callback for state exploration"""
        accepted, response = server.handle_message(msg)
        return response

    def verify_callback(msg: bytes) -> Tuple[bool, bytes]:
        """Callback for constraint verification"""
        return server.handle_message(msg)

    # Initialize NeuPRE
    neupre = NeuPRE(
        ib_d_model=128,
        ib_nhead=4,
        ib_layers=3,
        ib_beta=0.1,
        dkl_embedding_dim=128,
        dkl_hidden_dim=256,
        dkl_feature_dim=64,
        dkl_kappa=2.0,
        confidence_threshold=0.7,
        output_dir='./example_output'
    )

    # Run full pipeline
    logging.info("\nRunning NeuPRE pipeline...")

    results = neupre.run_full_pipeline(
        initial_messages=seed_messages,
        initial_responses=initial_responses,
        probe_callback=probe_callback,
        verify_callback=verify_callback,
        format_epochs=30,  # Reduced for example
        format_batch_size=16,
        exploration_iterations=50,  # Reduced for example
        exploration_mutations=30,
        enable_logic_refinement=True
    )

    # Display results
    logging.info("\n" + "=" * 80)
    logging.info("RESULTS")
    logging.info("=" * 80)

    # Phase 1 results
    logging.info("\nPhase 1: Format Learning")
    if results['phase1']['segmentations']:
        example_seg = results['phase1']['segmentations'][0]
        logging.info(f"  Example segmentation: {example_seg}")
        logging.info(f"  Number of fields detected: {len(example_seg) - 1}")

    # Phase 2 results
    if 'phase2' in results and results['phase2']:
        logging.info("\nPhase 2: State Exploration")
        logging.info(f"  Unique states discovered: {results['phase2'].get('unique_states', 'N/A')}")

    # Phase 3 results
    if 'phase3' in results and results['phase3']:
        logging.info("\nPhase 3: Logic Refinement")
        verified = results['phase3'].get('verified_rules', [])
        logging.info(f"  Verified rules: {len(verified)}")
        for rule in verified:
            logging.info(f"    Field {rule['field_index']}: {rule['field_type']} "
                       f"(confidence={rule['confidence']:.2f})")

    logging.info(f"\nTotal time: {results['total_time']:.2f}s")
    logging.info(f"Total server requests: {server.message_count}")

    # Export specification
    neupre.export_protocol_specification('simple_protocol_spec.txt')
    logging.info("\nProtocol specification exported to example_output/simple_protocol_spec.txt")

    # Save models
    neupre.save_models()
    logging.info("Models saved to example_output/")

    logging.info("\n" + "=" * 80)
    logging.info("Example completed successfully!")
    logging.info("=" * 80)


if __name__ == '__main__':
    main()
