import numpy as np
import logging
from typing import List

class HMMSegmenter:
    """
    基于隐马尔可夫模型(HMM)的协议结构推断器。
    状态：Field_Start, Inside_Field
    观测：NeuPRE 提取的互信息 (MI) 特征
    """
    def __init__(self, num_states=2):
        self.num_states = num_states
        # 简单两状态模型：0=Inside, 1=Start
        # 转移概率 (硬编码初始值，后期可学习)
        self.trans_prob = np.array([
            [0.8, 0.2], # Inside -> Inside (80%), Inside -> Start (20%)
            [0.1, 0.9]  # Start -> Inside (100% - Start 只能持续1字节)
        ])
        
    def segment(self, mi_scores: List[float]) -> List[int]:
        """
        使用 Viterbi 算法解码最佳分割路径
        :param mi_scores: 信息瓶颈模型输出的逐字节互信息 (Mutual Information)
        """
        T = len(mi_scores)
        if T == 0: return []
        
        # 1. 观测概率 (Emission Probability)
        # 高 MI -> 更有可能是 Field_Start
        # 低 MI -> 更有可能是 Inside_Field
        # 简单建模：MI > mean 是 Start
        mi_mean = np.mean(mi_scores) if len(mi_scores) > 0 else 0
        
        # dp[t][state] = max prob
        dp = np.zeros((T, 2))
        path = np.zeros((T, 2), dtype=int)
        
        # Init
        dp[0][0] = 0.5
        dp[0][1] = 0.5
        
        for t in range(1, T):
            score = mi_scores[t]
            # 观测似然 (Emission)
            # P(score | Inside) ~ 倾向于低分
            p_emit_0 = 1.0 / (1.0 + score) 
            # P(score | Start) ~ 倾向于高分
            p_emit_1 = score / (1.0 + score)
            
            for s in range(2): # Current state
                # Find best prev state
                probs = [dp[t-1][prev] * self.trans_prob[prev][s] for prev in range(2)]
                best_prev = np.argmax(probs)
                
                emit = p_emit_0 if s == 0 else p_emit_1
                dp[t][s] = probs[best_prev] * emit
                path[t][s] = best_prev
                
        # Backtrack
        boundaries = []
        curr = np.argmax(dp[T-1])
        for t in range(T-1, 0, -1):
            if curr == 1: # Field Start
                boundaries.append(t)
            curr = path[t][curr]
            
        return sorted(boundaries)