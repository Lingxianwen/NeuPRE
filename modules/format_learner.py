"""
Module 1: Information Bottleneck Based Format Learner

This module implements the Information Bottleneck principle for automatic field boundary detection.
Instead of using heuristic rules like DYNpre, it uses information compression theory to discover
optimal field boundaries.

Theory:
- View message X as input
- View "predict next byte" or "predict server response" as task Y
- Field partition T should maximize prediction power for Y while minimizing its own complexity

Optimization objective (IB Lagrangian):
    L_IB = min_T ( I(X; T) - β * I(T; Y) )

Where:
- I(X; T): Compression term - forces model to merge redundant bytes
- I(T; Y): Prediction term - forces model to keep critical bytes as separate fields
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging


class ProtocolDataset(Dataset):
    """Dataset for protocol messages"""

    def __init__(self, messages: List[bytes], responses: Optional[List[bytes]] = None,
                 max_length: int = 512):
        """
        Args:
            messages: List of protocol messages
            responses: Optional list of server responses
            max_length: Maximum message length
        """
        self.messages = messages
        self.responses = responses
        self.max_length = max_length

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        # Convert bytes to tensor
        msg = self.messages[idx]
        msg_tensor = torch.zeros(self.max_length, dtype=torch.long)
        msg_len = min(len(msg), self.max_length)
        msg_tensor[:msg_len] = torch.tensor([b for b in msg[:msg_len]], dtype=torch.long)

        if self.responses is not None:
            resp = self.responses[idx]
            resp_tensor = torch.zeros(self.max_length, dtype=torch.long)
            resp_len = min(len(resp), self.max_length)
            resp_tensor[:resp_len] = torch.tensor([b for b in resp[:resp_len]], dtype=torch.long)
            return msg_tensor, resp_tensor, msg_len, resp_len
        else:
            return msg_tensor, msg_len


class TransformerFieldEncoder(nn.Module):
    """
    Lightweight Transformer encoder for field extraction.
    Uses self-attention to automatically identify field boundaries.
    """

    def __init__(self, vocab_size: int = 256, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 3, dropout: float = 0.1):
        """
        Args:
            vocab_size: Vocabulary size (256 for bytes)
            d_model: Dimension of model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Boundary detection head
        self.boundary_head = nn.Linear(d_model, 2)  # Binary: boundary or not

        # Next byte prediction head (for self-supervised learning)
        self.prediction_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len)
            mask: Optional attention mask

        Returns:
            boundary_logits: Boundary predictions (batch, seq_len, 2)
            next_byte_logits: Next byte predictions (batch, seq_len, vocab_size)
            attention_weights: Attention weights for boundary extraction
        """
        # Embedding
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=mask)

        # Boundary prediction
        boundary_logits = self.boundary_head(encoded)

        # Next byte prediction
        next_byte_logits = self.prediction_head(encoded)

        # Extract attention weights (from last layer)
        attention_weights = None
        for layer in self.transformer.layers:
            if hasattr(layer, 'self_attn'):
                # Get attention weights from the last layer
                pass  # Will be extracted during attention computation

        return boundary_logits, next_byte_logits, encoded


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class InformationBottleneckFormatLearner:
    """
    Main class for Information Bottleneck based format learning.

    This is more robust than DYNpre's n-gram statistics, especially for:
    - Encrypted or high-entropy fields
    - Variable-length fields
    - Complex protocol structures
    """

    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 3,
                 beta: float = 0.1, learning_rate: float = 1e-3, device: str = 'cuda'):
        """
        Args:
            d_model: Dimension of transformer model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            beta: IB trade-off parameter (balances compression vs prediction)
            learning_rate: Learning rate for optimization
            device: Device to use (cuda/cpu)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.beta = beta
        self.learning_rate = learning_rate

        # Initialize model
        self.model = TransformerFieldEncoder(
            vocab_size=256,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        logging.info(f"InformationBottleneckFormatLearner initialized on {self.device}")
        logging.info(f"Model parameters: d_model={d_model}, nhead={nhead}, layers={num_layers}, beta={beta}")

    def get_mi_scores(self, message: bytes) -> List[float]:
        """
        新方法：仅返回互信息分数，不切分。
        这反映了每个字节包含的'惊奇度' (Surprisal)。
        """
        self.model.eval()
        with torch.no_grad():
            # ... (预处理代码，转为 tensor) ...
            src = ... # [1, len, d_model]
            
            # 获取 Encoder 输出 (BottleNeck 特征)
            z, _ = self.model.encoder(src) 
            
            # 计算相邻字节的特征距离作为 MI 代理
            # 距离越大 -> 突变 -> 可能是边界
            scores = []
            for i in range(1, z.size(1)):
                dist = torch.norm(z[0, i] - z[0, i-1]).item()
                scores.append(dist)
            
            # 补齐第一个
            scores.insert(0, 0.0)
            return scores
    
    def compute_ib_loss(self, encoded, boundary_logits, next_byte_logits,
                       next_bytes, lengths):
        """
        Compute Information Bottleneck loss.

        L_IB = I(X; T) - β * I(T; Y)

        Where:
        - I(X; T) is approximated by the entropy of boundary predictions (compression)
        - I(T; Y) is approximated by prediction accuracy (mutual information with task)
        """
        batch_size = encoded.size(0)

        # Compression term: I(X; T) - minimize entropy of field representations
        # Encourage sparse boundary predictions
        boundary_probs = F.softmax(boundary_logits, dim=-1)
        compression_loss = -torch.sum(boundary_probs * torch.log(boundary_probs + 1e-10)) / batch_size

        # Prediction term: I(T; Y) - maximize prediction accuracy
        # Self-supervised learning: predict next byte
        prediction_loss = F.cross_entropy(
            next_byte_logits[:, :-1, :].reshape(-1, 256),
            next_bytes[:, 1:].reshape(-1),
            ignore_index=0,  # Ignore padding
            reduction='mean'
        )

        # Information Bottleneck objective
        ib_loss = compression_loss - self.beta * (-prediction_loss)

        return ib_loss, compression_loss, prediction_loss

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_compression = 0
        total_prediction = 0

        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 2:  # No responses
                messages, lengths = batch
                messages = messages.to(self.device)
                next_bytes = messages  # Self-supervised: predict next byte in same message
            else:  # With responses
                messages, responses, msg_lengths, resp_lengths = batch
                messages = messages.to(self.device)
                responses = responses.to(self.device)
                next_bytes = responses  # Supervised: predict response bytes
                lengths = msg_lengths

            # Forward pass
            boundary_logits, next_byte_logits, encoded = self.model(messages)

            # Compute loss
            ib_loss, comp_loss, pred_loss = self.compute_ib_loss(
                encoded, boundary_logits, next_byte_logits, next_bytes, lengths
            )

            # Backward pass
            self.optimizer.zero_grad()
            ib_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += ib_loss.item()
            total_compression += comp_loss.item()
            total_prediction += pred_loss.item()

            if batch_idx % 10 == 0:
                logging.debug(f"Epoch {epoch} Batch {batch_idx}: "
                            f"Loss={ib_loss.item():.4f}, "
                            f"Compression={comp_loss.item():.4f}, "
                            f"Prediction={pred_loss.item():.4f}")

        n_batches = len(dataloader)
        return total_loss / n_batches, total_compression / n_batches, total_prediction / n_batches

    def train(self, messages: List[bytes], responses: Optional[List[bytes]] = None,
             epochs: int = 50, batch_size: int = 32, max_length: int = 512):
        """
        Train the model on protocol messages.

        Args:
            messages: List of protocol messages
            responses: Optional list of server responses
            epochs: Number of training epochs
            batch_size: Batch size
            max_length: Maximum message length
        """
        logging.info(f"Training on {len(messages)} messages for {epochs} epochs")

        # Create dataset and dataloader
        dataset = ProtocolDataset(messages, responses, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            loss, comp_loss, pred_loss = self.train_epoch(dataloader, epoch)

            if epoch % 5 == 0:
                logging.info(f"Epoch {epoch}/{epochs}: "
                           f"Loss={loss:.4f}, "
                           f"Compression={comp_loss:.4f}, "
                           f"Prediction={pred_loss:.4f}")

        logging.info("Training completed")

    def extract_boundaries(self, message: bytes, threshold: float = 0.5) -> List[int]:
        """
        Extract field boundaries from a message using attention weights.

        Args:
            message: Protocol message
            threshold: Threshold for boundary detection

        Returns:
            List of boundary positions
        """
        self.model.eval()

        # Prepare input
        msg_tensor = torch.zeros(len(message), dtype=torch.long)
        msg_tensor[:len(message)] = torch.tensor([b for b in message], dtype=torch.long)
        msg_tensor = msg_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            boundary_logits, _, _ = self.model(msg_tensor)
            boundary_probs = F.softmax(boundary_logits, dim=-1)

            # Extract boundary positions where probability > threshold
            boundaries = [0]  # Start of message
            for i in range(len(message)):
                if boundary_probs[0, i, 1].item() > threshold:
                    boundaries.append(i)
            boundaries.append(len(message))  # End of message

            # Remove duplicates and sort
            boundaries = sorted(list(set(boundaries)))

        return boundaries

    def segment_message(self, message: bytes, threshold: float = 0.5) -> List[Tuple[int, int]]:
        """
        Segment message into fields.

        Args:
            message: Protocol message
            threshold: Threshold for boundary detection

        Returns:
            List of (start, end) tuples for each field
        """
        boundaries = self.extract_boundaries(message, threshold)
        segments = []

        for i in range(len(boundaries) - 1):
            segments.append((boundaries[i], boundaries[i + 1]))

        return segments

    def save_model(self, path: str):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'beta': self.beta
        }, path)
        logging.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.beta = checkpoint['beta']
        logging.info(f"Model loaded from {path}")


def compare_with_dynpre(ib_boundaries: List[int], dynpre_boundaries: List[int],
                        ground_truth: List[int]) -> Dict[str, float]:
    """
    Compare IB-based segmentation with DYNpre segmentation.

    Args:
        ib_boundaries: Boundaries from Information Bottleneck
        dynpre_boundaries: Boundaries from DYNpre
        ground_truth: Ground truth boundaries

    Returns:
        Dictionary with comparison metrics
    """
    def compute_metrics(pred, gt):
        pred_set = set(pred)
        gt_set = set(gt)

        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Perfect match
        perfect = 1.0 if pred_set == gt_set else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'perfect': perfect
        }

    ib_metrics = compute_metrics(ib_boundaries, ground_truth)
    dynpre_metrics = compute_metrics(dynpre_boundaries, ground_truth)

    return {
        'ib': ib_metrics,
        'dynpre': dynpre_metrics
    }
