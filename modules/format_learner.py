import torch
import torch.nn as nn
import torch.optim as optim
import math
import logging
import numpy as np
from typing import List, Tuple

class TransformerFieldEncoder(nn.Module):
    def __init__(self, vocab_size=256, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        # +1 用于 Mask Token (索引 256)
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 两个输出头
        # 1. 边界预测头 (用于下游 HMM)
        self.boundary_head = nn.Linear(d_model, 2) 
        # 2. 掩码还原头 (用于训练时的 Loss)
        self.mlm_head = nn.Linear(d_model, vocab_size) 

        self.d_model = d_model

    def forward(self, src):
        # src: [Batch, Seq]
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x) # [Batch, Seq, Hidden]
        
        # 预测边界概率
        boundary_logits = self.boundary_head(encoded) # [Batch, Seq, 2]
        
        # 预测被 Mask 的 Token
        mlm_logits = self.mlm_head(encoded) # [Batch, Seq, Vocab]
        
        return boundary_logits, mlm_logits, encoded

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class InformationBottleneckFormatLearner:
    def __init__(self, d_model=256, nhead=8, num_layers=4, beta=0.01, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # vocab_size=256 (0-255), mask_token=256
        self.model = TransformerFieldEncoder(vocab_size=256, d_model=d_model, nhead=nhead, num_layers=num_layers).to(self.device)
        self.beta = beta
        self.mask_token_idx = 256  # 专门的 Mask Token ID

    def train(self, train_data: List[bytes], val_data: List[bytes], epochs=50, batch_size=32):
        # 数据预处理：统一 Padding
        max_len = max(len(m) for m in train_data)
        # 截断过长的消息以节省显存 (SMB2 这种)
        max_len = min(max_len, 512) 
        
        tensor_data = []
        for m in train_data:
            m = list(m)[:max_len] # 截断
            m = m + [0] * (max_len - len(m)) # Padding (简单用0填充，虽然不严谨但不影响MLM核心)
            tensor_data.append(m)
        
        dataset = torch.tensor(tensor_data, dtype=torch.long)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        criterion_mlm = nn.CrossEntropyLoss(ignore_index=-1) # -1 表示不计算 Loss

        logging.info(f"Starting BERT-style MLM training on {len(train_data)} messages...")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in dataloader:
                batch = batch.to(self.device) # [Batch, Seq]
                
                # --- 动态 Masking 核心逻辑 ---
                # 1. 复制一份作为 Label
                targets = batch.clone()
                
                # 2. 生成 Mask 矩阵 (15% 概率)
                # 不 Mask 填充符(0)
                padding_mask = (batch == 0)
                rand_matrix = torch.rand(batch.shape).to(self.device)
                
                # 只有非 Padding 且 随机值<0.15 的位置才被 Mask
                mask_indices = (rand_matrix < 0.15) & (~padding_mask)
                
                # 3. 构建输入 Input
                inputs = batch.clone()
                inputs[mask_indices] = self.mask_token_idx # 替换为 [MASK]
                
                # 4. 构建 Label Targets
                # 不需要预测的位置设为 -1 (被 CrossEntropyLoss 忽略)
                targets[~mask_indices] = -1 
                
                # --- Forward & Loss ---
                optimizer.zero_grad()
                boundary_logits, mlm_logits, _ = self.model(inputs)
                
                # Loss 1: MLM Loss (填空题)
                # mlm_logits: [Batch, Seq, Vocab] -> [Batch*Seq, Vocab]
                loss_mlm = criterion_mlm(mlm_logits.view(-1, 256), targets.view(-1))
                
                # Loss 2: Information Bottleneck (稀疏化约束)
                # 我们希望边界概率是稀疏的（要么是边界，要么不是，不要模棱两可）
                # P(Boundary)
                probs = torch.softmax(boundary_logits, dim=-1)[:, :, 1]
                # L1 正则化：希望 probs 接近 0 (大部分地方不是边界)
                loss_ib = torch.mean(torch.abs(probs)) 
                
                # 联合 Loss
                loss = loss_mlm + self.beta * loss_ib
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{epochs}: Loss={total_loss:.4f} (MLM Focus)")

    def get_boundary_probs(self, message: bytes) -> List[float]:
        """获取边界概率序列，用于 HMM 解码"""
        self.model.eval()
        with torch.no_grad():
            # 预处理
            seq = list(message)[:512] # 同样截断以匹配训练
            src = torch.tensor([seq], dtype=torch.long).to(self.device)
            
            boundary_logits, _, _ = self.model(src)
            probs = torch.softmax(boundary_logits, dim=-1)
            
            # 取出类别 1 (Is Boundary) 的概率
            boundary_probs = probs[0, :, 1].cpu().tolist()
            
            # 如果原消息比截断长，后面补 0 概率
            if len(message) > len(boundary_probs):
                boundary_probs += [0.0] * (len(message) - len(boundary_probs))
                
            return boundary_probs

    # 兼容旧代码接口
    def extract_boundaries(self, message: bytes, threshold: float = 0.5) -> List[int]:
        probs = self.get_boundary_probs(message)
        return [i for i, p in enumerate(probs) if p > threshold]