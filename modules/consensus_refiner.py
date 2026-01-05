import numpy as np
import logging
from typing import List

class StatisticalConsensusRefiner:
    """
    基于群体统计的后处理器。
    原理：真正的协议字段边界在不同消息中通常是对齐的。
    """
    def __init__(self, 
                 min_support: float = 0.4,  # 至少 40% 的消息支持该边界
                 tolerance: int = 0):       # 容忍 0 字节的误差
        self.min_support = min_support
        self.tolerance = tolerance

    def refine(self, messages: List[bytes], raw_boundaries: List[List[int]]) -> List[List[int]]:
        if not raw_boundaries:
            return raw_boundaries

        num_msgs = len(messages)
        # 找到最长消息长度
        max_len = max(len(m) for m in messages) if messages else 0
        
        # 1. 构建边界热力图 (Boundary Heatmap)
        # boundary_votes[i] 表示有多少个消息在索引 i 处有边界
        boundary_votes = np.zeros(max_len + 1, dtype=int)
        
        # 统计有效投票的分母 (在该位置有数据的消息数量)
        valid_counts = np.zeros(max_len + 1, dtype=int)

        for i, msg in enumerate(messages):
            msg_len = len(msg)
            # 记录该消息覆盖的长度范围
            valid_counts[:msg_len+1] += 1
            
            # 投票
            for b in raw_boundaries[i]:
                if b <= max_len:
                    boundary_votes[b] += 1

        # 2. 筛选强边界 (Global Strong Boundaries)
        strong_boundaries = set()
        # 总是包含 0
        strong_boundaries.add(0)
        
        for idx in range(1, max_len):
            if valid_counts[idx] < (num_msgs * 0.1): # 忽略太深层、样本太少的位置
                continue
                
            support = boundary_votes[idx] / valid_counts[idx]
            
            # 核心逻辑：如果支持度超过阈值，或者即使支持度低但在该局部区域是峰值
            if support >= self.min_support:
                strong_boundaries.add(idx)

        logging.info(f"Refiner found {len(strong_boundaries)} consensus boundaries: {sorted(list(strong_boundaries))}")

        # 3. 将强边界应用回每个消息 (Projection)
        refined_results = []
        for i, msg in enumerate(messages):
            msg_len = len(msg)
            # 基础边界：0 和 结尾
            final_bounds = {0, msg_len}
            
            # 添加适用的强边界
            for b in strong_boundaries:
                if b < msg_len:
                    final_bounds.add(b)
            
            # (可选) 保留部分原始的高置信度边界？
            # 这里我们选择“严格模式”：只保留共识边界，以大幅提高 Precision
            
            refined_results.append(sorted(list(final_bounds)))

        return refined_results