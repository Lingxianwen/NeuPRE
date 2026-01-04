"""
Improved NeuPRE Format Learner with Supervised Learning

This version uses ground truth boundaries for supervised training,
dramatically improving segmentation accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Tuple
import numpy as np


class SupervisedFormatLearner:
    """
    Supervised format learner that uses ground truth boundaries for training.

    This is a simplified but more effective approach compared to the
    unsupervised Information Bottleneck method.
    """

    def __init__(self, d_model=128, nhead=4, num_layers=2, device='cuda'):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Simple CNN-based boundary detector
        self.model = nn.Sequential(
            # Input: (batch, seq_len, 256) - one-hot encoded bytes
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 2, kernel_size=1),  # Output: 2 classes (boundary or not)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        logging.info(f"SupervisedFormatLearner initialized on {self.device}")

    def train(self, messages: List[bytes], ground_truth: List[List[int]], epochs=50, batch_size=16):
        """
        Train the model using ground truth boundaries.

        Args:
            messages: List of protocol messages
            ground_truth: List of boundary lists (ground truth)
            epochs: Number of training epochs
            batch_size: Batch size
        """
        logging.info(f"Training on {len(messages)} messages for {epochs} epochs")

        # Prepare training data
        max_len = max(len(msg) for msg in messages)

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for i in range(0, len(messages), batch_size):
                batch_messages = messages[i:i+batch_size]
                batch_gt = ground_truth[i:i+batch_size]

                # Encode messages as one-hot
                batch_encoded = []
                batch_labels = []

                for msg, gt in zip(batch_messages, batch_gt):
                    # One-hot encode
                    encoded = torch.zeros(max_len, 256)
                    for j, byte_val in enumerate(msg):
                        if j < max_len:
                            encoded[j, byte_val] = 1.0

                    # Create labels (1 for boundary, 0 for non-boundary)
                    labels = torch.zeros(max_len, dtype=torch.long)
                    for boundary_idx in gt:
                        if boundary_idx < max_len:
                            labels[boundary_idx] = 1

                    batch_encoded.append(encoded)
                    batch_labels.append(labels)

                # Stack into batch
                batch_tensor = torch.stack(batch_encoded).to(self.device)  # (batch, seq_len, 256)
                label_tensor = torch.stack(batch_labels).to(self.device)   # (batch, seq_len)

                # Forward pass
                # Conv1d expects (batch, channels, seq_len)
                batch_tensor = batch_tensor.transpose(1, 2)  # (batch, 256, seq_len)
                outputs = self.model(batch_tensor)            # (batch, 2, seq_len)
                outputs = outputs.transpose(1, 2)             # (batch, seq_len, 2)

                # Compute loss
                loss = F.cross_entropy(
                    outputs.reshape(-1, 2),
                    label_tensor.reshape(-1),
                    weight=torch.tensor([1.0, 10.0]).to(self.device)  # Higher weight for boundaries
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            if epoch % 10 == 0 or epoch == epochs - 1:
                logging.info(f"Epoch {epoch}/{epochs}: Loss={avg_loss:.4f}")

        logging.info("Training completed")

    def extract_boundaries(self, message: bytes, threshold=0.5) -> List[int]:
        """
        Extract field boundaries from a message.

        Args:
            message: Protocol message
            threshold: Probability threshold for boundary detection

        Returns:
            List of boundary positions
        """
        self.model.eval()

        # One-hot encode
        encoded = torch.zeros(len(message), 256)
        for i, byte_val in enumerate(message):
            encoded[i, byte_val] = 1.0

        # Add batch dimension and transpose
        input_tensor = encoded.unsqueeze(0).transpose(1, 2).to(self.device)  # (1, 256, seq_len)

        with torch.no_grad():
            outputs = self.model(input_tensor)  # (1, 2, seq_len)
            probs = F.softmax(outputs, dim=1)   # (1, 2, seq_len)
            boundary_probs = probs[0, 1, :]      # (seq_len,) - probability of being a boundary

        # Extract boundaries
        boundaries = [0]  # Always start with 0
        for i in range(len(message)):
            if boundary_probs[i].item() > threshold and i > 0:
                boundaries.append(i)

        # Always end with message length
        if boundaries[-1] != len(message):
            boundaries.append(len(message))

        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))

        return boundaries


def test_supervised_learner():
    """Test supervised learner on synthetic data."""
    from experiments.experiment2_segmentation import ProtocolDataset

    # Generate synthetic data
    messages, ground_truth = ProtocolDataset.generate_simple_protocol(100)

    # Split into train/test
    train_messages = messages[:80]
    train_gt = ground_truth[:80]
    test_messages = messages[80:]
    test_gt = ground_truth[80:]

    # Train
    learner = SupervisedFormatLearner(d_model=128, nhead=4, num_layers=2)
    learner.train(train_messages, train_gt, epochs=50, batch_size=16)

    # Test
    print("\nTesting on unseen messages:")
    for i in range(5):
        msg = test_messages[i]
        gt = test_gt[i]
        pred = learner.extract_boundaries(msg, threshold=0.5)

        print(f"\nMessage {i+1}: {msg.hex()}")
        print(f"  Ground truth: {gt}")
        print(f"  Predicted:    {pred}")

        # Compute accuracy
        gt_set = set(gt)
        pred_set = set(pred)
        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_supervised_learner()