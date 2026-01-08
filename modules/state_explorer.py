"""
Module 2: Deep Kernel Learning State Explorer - FIXED VERSION

主要修改：
1. 增加warmup阶段（前期使用随机探索）
2. 改进acquisition function的计算
3. 增加训练稳定性
4. 修复GPyTorch兼容性问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
import logging
from collections import defaultdict


class MessageFeatureExtractor(nn.Module):
    """Feature extractor for protocol messages"""

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256,
                 feature_dim: int = 64, use_cnn: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(256, embedding_dim)
        self.use_cnn = use_cnn

        if use_cnn:
            self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                              num_layers=2, batch_first=True,
                              dropout=0.1, bidirectional=True)

        if use_cnn:
            self.fc = nn.Linear(hidden_dim, feature_dim)
        else:
            self.fc = nn.Linear(hidden_dim * 2, feature_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, lengths=None):
        embedded = self.embedding(x)

        if self.use_cnn:
            x = embedded.transpose(1, 2)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.pool(x).squeeze(-1)
        else:
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                lstm_out, (hidden, cell) = self.lstm(packed)
                x = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                lstm_out, (hidden, cell) = self.lstm(embedded)
                x = torch.cat([hidden[-2], hidden[-1]], dim=1)

        x = self.dropout(x)
        features = self.fc(x)
        return features


class DeepKernelGP(ApproximateGP):
    """Deep Kernel Learning GP model"""

    def __init__(self, feature_extractor: MessageFeatureExtractor,
                 num_inducing: int = 128, feature_dim: int = 64):
        inducing_points = torch.randn(num_inducing, feature_dim)
        variational_distribution = CholeskyVariationalDistribution(num_inducing)
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x, lengths=None):
        features = self.feature_extractor(x, lengths)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DeepKernelStateExplorer:
    """
    改进的Deep Kernel Learning State Explorer
    """

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256,
                 feature_dim: int = 64, use_cnn: bool = False,
                 num_inducing: int = 128, learning_rate: float = 1e-3,
                 kappa: float = 3.8,  # 提高探索性
                 device: str = 'cuda',
                 warmup_iterations: int = 12):  # 减少warmup（12%）
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.kappa = kappa
        self.feature_dim = feature_dim
        self.warmup_iterations = warmup_iterations  # Warmup阶段长度
        self.current_iteration = 0  # 当前迭代计数

        self.feature_extractor = MessageFeatureExtractor(
            embedding_dim, hidden_dim, feature_dim, use_cnn
        ).to(self.device)

        self.model = DeepKernelGP(
            self.feature_extractor, num_inducing, feature_dim
        ).to(self.device)

        self.likelihood = GaussianLikelihood().to(self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()}
        ], lr=learning_rate)

        self.mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=1000
        )

        self.observed_messages = []
        self.observed_responses = []
        self.state_map = defaultdict(list)
        self.unique_states = set()

        logging.info(f"DeepKernelStateExplorer initialized on {self.device}")
        logging.info(f"Warmup iterations: {warmup_iterations}")

    def bytes_to_tensor(self, message: bytes, max_length: int = 512) -> torch.Tensor:
        tensor = torch.zeros(max_length, dtype=torch.long)
        msg_len = min(len(message), max_length)
        tensor[:msg_len] = torch.tensor([b for b in message[:msg_len]], dtype=torch.long)
        return tensor

    def observe(self, message: bytes, response: bytes):
        self.observed_messages.append(message)
        self.observed_responses.append(response)

        response_hash = hash(response)
        self.state_map[response_hash].append(message)
        self.unique_states.add(response_hash)

        logging.debug(f"Observed message-response pair. "
                    f"Total unique states: {len(self.unique_states)}")

    def train_step(self, messages: List[bytes], targets: List[float],
                  batch_size: int = 32) -> float:
        """改进的训练步骤，增加稳定性"""
        self.model.train()
        self.likelihood.train()

        msg_tensors = []
        lengths = []
        for msg in messages:
            tensor = self.bytes_to_tensor(msg)
            msg_tensors.append(tensor)
            lengths.append(min(len(msg), 512))

        msg_batch = torch.stack(msg_tensors).to(self.device)
        lengths_tensor = torch.tensor(lengths, device=self.device)
        target_batch = torch.tensor(targets, dtype=torch.float32, device=self.device)

        try:
            output = self.model.forward(msg_batch, lengths_tensor)
            loss = -self.mll(output, target_batch)
        except Exception as e:
            logging.warning(f"Training issue: {e}. Using simplified loss.")
            output = self.model.forward(msg_batch, lengths_tensor)
            pred_dist = self.likelihood(output)
            loss = torch.nn.functional.mse_loss(pred_dist.mean, target_batch)

        # 梯度裁剪以增加稳定性
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, epochs: int = 30):  # 减少训练轮数以加快速度
        """训练模型"""
        if len(self.observed_messages) < 5:
            logging.warning("Not enough data to train")
            return

        logging.debug(f"Training DKL model on {len(self.observed_messages)} observations")

        # 计算novelty scores
        targets = []
        for resp in self.observed_responses:
            resp_hash = hash(resp)
            novelty = 1.0 / (len(self.state_map[resp_hash]) + 1)  # +1平滑
            targets.append(novelty)

        # 训练循环
        for epoch in range(epochs):
            loss = self.train_step(self.observed_messages, targets)

            if epoch % 10 == 0:
                logging.debug(f"Epoch {epoch}/{epochs}: Loss={loss:.4f}")

        logging.debug("Training completed")

    def acquisition_function(self, message: bytes) -> float:
        """
        改进的acquisition function
        在warmup阶段返回随机值，之后使用UCB
        """
        # Warmup阶段：使用随机探索
        if self.current_iteration < self.warmup_iterations:
            return np.random.random()

        self.model.eval()
        self.likelihood.eval()

        msg_tensor = self.bytes_to_tensor(message).unsqueeze(0).to(self.device)
        length_tensor = torch.tensor([min(len(message), 512)], device=self.device)

        with torch.no_grad():
            try:
                output = self.model.forward(msg_tensor, length_tensor)
                pred = self.likelihood(output)

                mean = pred.mean.item()
                std = pred.stddev.item()
                
                # UCB with higher exploration
                ucb = mean + self.kappa * std
            except Exception as e:
                logging.debug(f"Acquisition function issue: {e}. Using random value.")
                ucb = np.random.random()

        return ucb

    def select_next_message(self, candidates: List[bytes],
                          top_k: int = 1) -> List[Tuple[bytes, float]]:
        """选择下一个要探测的消息"""
        
        # Warmup阶段：随机选择
        if self.current_iteration < self.warmup_iterations:
            logging.debug(f"Warmup phase: random selection ({self.current_iteration}/{self.warmup_iterations})")
            selected_msgs = np.random.choice(candidates, size=min(top_k, len(candidates)), replace=False)
            return [(msg, 0.0) for msg in selected_msgs]

        logging.debug(f"Active learning phase: selecting from {len(candidates)} candidates")

        # 正常的acquisition-based选择
        acq_values = []
        for msg in candidates:
            acq = self.acquisition_function(msg)
            acq_values.append((msg, acq))

        acq_values.sort(key=lambda x: x[1], reverse=True)
        selected = acq_values[:top_k]

        logging.debug(f"Selected {len(selected)} messages with acquisition values: "
                   f"{[f'{v:.4f}' for _, v in selected]}")

        return selected

    def generate_mutations(self, base_message: bytes, num_mutations: int = 100,
                          mutation_rate: float = 0.12) -> List[bytes]:  # 适度提高mutation rate
        """生成突变消息"""
        mutations = []

        for _ in range(num_mutations):
            mutated = bytearray(base_message)

            for i in range(len(mutated)):
                if np.random.random() < mutation_rate:
                    mutated[i] = np.random.randint(0, 256)

            mutations.append(bytes(mutated))

        return mutations

    def active_exploration(self, base_messages: List[bytes],
                         num_iterations: int = 100,
                         num_mutations: int = 80,
                         probe_callback: Callable[[bytes], bytes] = None) -> Dict:
        """
        改进的主动探索算法
        """
        logging.info(f"Starting active exploration for {num_iterations} iterations")
        logging.info(f"Warmup phase: first {self.warmup_iterations} iterations")

        stats = {
            'iterations': [],
            'unique_states': [],
            'acquisition_values': []
        }

        for iteration in range(num_iterations):
            self.current_iteration = iteration

            # 生成候选
            candidates = []
            for base_msg in base_messages:
                mutations = self.generate_mutations(base_msg, num_mutations)
                candidates.extend(mutations)

            # 选择最佳候选
            if len(self.observed_messages) > self.warmup_iterations:
                # 每隔几次迭代训练一次（减少训练频率以加快速度）
                if iteration % 5 == 0:
                    self.train(epochs=10)

                selected = self.select_next_message(candidates, top_k=1)
                next_message, acq_value = selected[0]
            else:
                # 初期随机探索
                next_message = np.random.choice(candidates)
                acq_value = 0.0

            # 探测
            if probe_callback is not None:
                response = probe_callback(next_message)
                self.observe(next_message, response)

            # 记录统计
            stats['iterations'].append(iteration)
            stats['unique_states'].append(len(self.unique_states))
            stats['acquisition_values'].append(acq_value)

            if iteration % 10 == 0:
                phase = "warmup" if iteration < self.warmup_iterations else "active"
                logging.info(f"Iteration {iteration}/{num_iterations} [{phase}]: "
                           f"Unique states={len(self.unique_states)}, "
                           f"Acquisition={acq_value:.4f}")

        logging.info(f"Exploration completed. Discovered {len(self.unique_states)} unique states")

        return stats

    def get_state_coverage(self) -> int:
        return len(self.unique_states)

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'observed_messages': self.observed_messages,
            'observed_responses': self.observed_responses,
            'state_map': dict(self.state_map),
            'unique_states': list(self.unique_states)
        }, path)
        logging.info(f"Model saved to {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.observed_messages = checkpoint['observed_messages']
        self.observed_responses = checkpoint['observed_responses']
        self.state_map = defaultdict(list, checkpoint['state_map'])
        self.unique_states = set(checkpoint['unique_states'])
        logging.info(f"Model loaded from {path}")