import numpy as np
from typing import List

class HMMSegmenter:
    def __init__(self):
        # 状态 0: Inside Field (字段内部)
        # 状态 1: Field Start (字段开始/边界)
        
        # [激进策略] 提高从 Inside -> Start 的转移概率
        # 这意味着模型倾向于认为字段比较短，经常发生跳变
        # Modbus 字段很多都是 1-2 字节，这很有帮助
        self.trans_prob = np.log(np.array([
            [0.55, 0.45],  # Inside -> Inside (55%), Inside -> Start (45%) <--- 调高这里
            [0.90, 0.10]   # Start -> Inside (90%), Start -> Start (10%)
        ]) + 1e-10)
        
        self.start_prob = np.log(np.array([0.5, 0.5]) + 1e-10)

    def segment(self, boundary_probs: List[float]) -> List[int]:
        T = len(boundary_probs)
        if T == 0: return []
        
        dp = np.zeros((T, 2))
        path = np.zeros((T, 2), dtype=int)
        
        # 初始化
        p = boundary_probs[0]
        # 发射概率加权：稍微放大神经网络的信号
        scale = 2.0 
        
        dp[0][0] = self.start_prob[0] + scale * np.log((1 - p) + 1e-10)
        dp[0][1] = self.start_prob[1] + scale * np.log(p + 1e-10)
        
        for t in range(1, T):
            p = boundary_probs[t]
            # 观测概率
            emit_0 = scale * np.log((1 - p) + 1e-10)
            emit_1 = scale * np.log(p + 1e-10)
            
            # Viterbi 递推
            for s in range(2):
                # 从上一时刻的哪个状态转移过来概率最大？
                prob_from_0 = dp[t-1][0] + self.trans_prob[0][s]
                prob_from_1 = dp[t-1][1] + self.trans_prob[1][s]
                
                if prob_from_0 > prob_from_1:
                    dp[t][s] = prob_from_0 + (emit_0 if s==0 else emit_1)
                    path[t][s] = 0
                else:
                    dp[t][s] = prob_from_1 + (emit_0 if s==0 else emit_1)
                    path[t][s] = 1
                    
        # 回溯
        boundaries = []
        curr = np.argmax(dp[T-1])
        for t in range(T-1, 0, -1):
            if curr == 1:
                boundaries.append(t)
            curr = path[t][curr]
            
        return sorted(boundaries)