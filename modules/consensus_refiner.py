import numpy as np
import logging
from typing import List

class StatisticalConsensusRefiner:
    """
    NeuPRE 核心增强模块：基于群体统计的边界精细化。
    原理：真正的协议字段边界在不同消息中通常是对齐的。
    单条消息的预测可能有噪声，但统计 100 条消息的投票结果能过滤噪声。
    """
    def __init__(self, 
                 min_support: float = 0.3,  # 只要有 30% 的消息认为这里是边界，就保留它
                 tolerance: int = 0):
        self.min_support = min_support
        self.tolerance = tolerance

    def refine(self, messages: List[bytes], raw_boundaries: List[List[int]]) -> List[List[int]]:
        if not raw_boundaries:
            return raw_boundaries

        # 1. 构建全局投票箱
        max_len = max(len(m) for m in messages) if messages else 0
        boundary_votes = np.zeros(max_len + 1, dtype=int)
        valid_counts = np.zeros(max_len + 1, dtype=int)

        # 2. 收集每一条消息的投票
        for i, msg in enumerate(messages):
            msg_len = len(msg)
            valid_counts[:msg_len+1] += 1
            for b in raw_boundaries[i]:
                if b <= max_len:
                    boundary_votes[b] += 1

        # 3. 筛选“强边界” (Consensus Boundaries)
        strong_boundaries = set()
        strong_boundaries.add(0) # 开头永远是边界
        
        logging.info("Analyzing consensus boundaries...")
        for idx in range(1, max_len):
            if valid_counts[idx] < (len(messages) * 0.1): continue # 忽略样本太少的区域
            
            support = boundary_votes[idx] / valid_counts[idx]
            
            # 核心逻辑：支持度超过阈值
            if support >= self.min_support:
                strong_boundaries.add(idx)
        
        sorted_bounds = sorted(list(strong_boundaries))
        logging.info(f"Refiner found {len(sorted_bounds)} consensus boundaries: {sorted_bounds}")

        # 4. 将强边界应用回每一条消息 (Projection)
        refined_results = []
        for i, msg in enumerate(messages):
            msg_len = len(msg)
            # 基础边界：0 和 结尾
            final_bounds = {0, msg_len}
            
            # 只有当强边界在消息长度范围内时才添加
            for b in strong_boundaries:
                if b < msg_len:
                    final_bounds.add(b)
            
            refined_results.append(sorted(list(final_bounds)))

        return refined_results