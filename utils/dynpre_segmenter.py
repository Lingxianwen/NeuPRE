"""
Real DynPRE Segmentation Implementation

Based on the original DynPRE source code, using Netzob for initial alignment-based splitting.
"""

import logging
from typing import List, Tuple
import numpy as np

try:
    from netzob.all import Format, RawMessage, Symbol
    NETZOB_AVAILABLE = True
except ImportError:
    NETZOB_AVAILABLE = False
    logging.warning("Netzob not available. Install with: pip install netzob")


class DynPRESegmenter:
    """
    Real DynPRE segmentation algorithm based on the original implementation.

    Uses Netzob's Format.splitAligned() for alignment-based message format inference.
    """

    def __init__(self):
        if not NETZOB_AVAILABLE:
            raise ImportError("Netzob is required for DynPRE segmentation. Install with: pip install netzob")

    def segment_messages(self, messages: List[bytes]) -> List[List[int]]:
        """
        Segment messages using DynPRE's alignment-based approach.

        Args:
            messages: List of protocol messages (bytes)

        Returns:
            List of boundary lists for each message
        """
        if len(messages) == 0:
            return []

        # Convert bytes to Netzob RawMessage objects
        netzob_messages = [RawMessage(msg) for msg in messages]

        # Create a symbol (protocol format representation)
        symbol = Symbol(messages=netzob_messages)

        # Use Netzob's alignment-based splitting
        # This is the core of DynPRE's initial segmentation
        Format.splitAligned(symbol, doInternalSlick=False)

        # Extract segmentation boundaries for each message
        message_cells = symbol.getMessageCells()
        segmentations = []

        for msg in symbol.messages:
            boundaries = [0]
            current_pos = 0

            # Get fields for this message
            fields = message_cells[msg]

            for field in fields:
                field_len = len(field.hex()) // 2  # Convert hex string length to byte length
                if field_len > 0:
                    current_pos += field_len
                    boundaries.append(current_pos)

            # Ensure we have end boundary
            if boundaries[-1] != len(msg.data):
                boundaries.append(len(msg.data))

            segmentations.append(boundaries)

        return segmentations

    def segment_with_clustering(self, messages: List[bytes]) -> List[List[int]]:
        """
        Segment messages using DynPRE's clustering approach.

        This is more sophisticated - it clusters messages by similarity first.

        Args:
            messages: List of protocol messages (bytes)

        Returns:
            List of boundary lists for each message
        """
        if len(messages) == 0:
            return []

        # Convert to Netzob messages
        netzob_messages = [RawMessage(msg) for msg in messages]

        # Cluster messages by alignment similarity
        # This groups similar messages together before splitting
        symbols = Format.clusterByAlignment(messages=netzob_messages, internalSlick=True)

        logging.info(f"DynPRE clustered {len(messages)} messages into {len(symbols)} format clusters")

        # Map each message to its boundaries
        msg_to_boundaries = {}

        for symbol in symbols:
            message_cells = symbol.getMessageCells()

            for msg in symbol.messages:
                boundaries = [0]
                current_pos = 0

                fields = message_cells[msg]
                for field in fields:
                    field_len = len(field.hex()) // 2
                    if field_len > 0:
                        current_pos += field_len
                        boundaries.append(current_pos)

                if boundaries[-1] != len(msg.data):
                    boundaries.append(len(msg.data))

                # Store using message hash
                msg_to_boundaries[hash(msg.data.hex())] = boundaries

        # Extract boundaries in original order
        segmentations = []
        for msg in messages:
            msg_hash = hash(msg.hex())
            if msg_hash in msg_to_boundaries:
                segmentations.append(msg_to_boundaries[msg_hash])
            else:
                # Fallback: just start and end
                segmentations.append([0, len(msg)])

        return segmentations


def test_dynpre_segmenter():
    """Test DynPRE segmenter on sample data."""
    if not NETZOB_AVAILABLE:
        print("Netzob not available. Skipping test.")
        return

    # Sample Modbus TCP messages (same as in ground truth)
    messages = [
        bytes.fromhex('000100000006ff050000ff00'),
        bytes.fromhex('000100000006ff050000ff00'),
        bytes.fromhex('000200000006ff0100000001'),
        bytes.fromhex('000300000006ff0100010001'),
        bytes.fromhex('000400000006ff0500010000'),
    ]

    segmenter = DynPRESegmenter()

    print("Testing DynPRE segmentation (alignment-based):")
    segmentations = segmenter.segment_messages(messages)

    for i, (msg, seg) in enumerate(zip(messages, segmentations)):
        print(f"\nMessage {i+1}: {msg.hex()}")
        print(f"  Boundaries: {seg}")
        print(f"  Fields: {len(seg)-1}")

    print("\n" + "="*80)
    print("Testing DynPRE segmentation (with clustering):")
    segmentations_cluster = segmenter.segment_with_clustering(messages)

    for i, (msg, seg) in enumerate(zip(messages, segmentations_cluster)):
        print(f"\nMessage {i+1}: {msg.hex()}")
        print(f"  Boundaries: {seg}")
        print(f"  Fields: {len(seg)-1}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_dynpre_segmenter()