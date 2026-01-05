import numpy as np
from typing import List, Dict
import logging

class GlobalAlignmentRefiner:
    """
    NeuPRE 后处理器：利用全局对齐信息精细化分割结果。
    模拟 DynPRE 的 Refinement 阶段，但基于统计学投票。
    """
    def __init__(self, alignment_threshold: float = 0.6):
        """
        :param alignment_threshold: 如果超过 60% 的消息在位置 X 有边界，则强制所有消息在 X 切分
        """
        self.threshold = alignment_threshold

    def refine(self, messages: List[bytes], boundaries: List[List[int]]) -> List[List[int]]:
        if not messages or not boundaries:
            return boundaries
            
        num_msgs = len(messages)
        max_len = max(len(m) for m in messages)
        
        # 1. 统计全局边界频率 (Global Boundary Frequency)
        # 统计每个字节索引处出现边界的次数
        boundary_counts = {}
        for b_list in boundaries:
            for idx in b_list:
                boundary_counts[idx] = boundary_counts.get(idx, 0) + 1
                
        # 2. 识别“强边界” (Strong Boundaries)
        # 只有当某个位置在足够多的消息中都被标记为边界时，才认为它是真正的协议字段边界
        strong_boundaries = set()
        for idx, count in boundary_counts.items():
            # 忽略 0 和结尾，只看中间
            if idx == 0: continue
            
            # 计算频率：注意这里分母应该是“长度足够覆盖该索引的消息总数”
            # 简单起见，我们用总消息数作为基数（假设大部分消息长度相近，如 Modbus）
            valid_msgs_count = sum(1 for m in messages if len(m) >= idx)
            if valid_msgs_count == 0: continue
            
            frequency = count / valid_msgs_count
            if frequency >= self.threshold:
                strong_boundaries.add(idx)
        
        logging.info(f"Refinement identified {len(strong_boundaries)} strong global boundaries: {sorted(list(strong_boundaries))}")

        # 3. 应用强边界到所有消息 (Enforce Alignment)
        refined_segmentations = []
        for i, msg in enumerate(messages):
            # 获取原始边界
            current_bounds = set(boundaries[i])
            
            # 强制添加强边界 (如果该消息长度允许)
            for sb in strong_boundaries:
                if sb < len(msg):
                    current_bounds.add(sb)
            
            # 4. (可选) 过滤掉过于碎片的边界
            # 如果两个边界距离太近 (例如 1 字节)，且不是强边界，可以考虑移除
            # 这里先不做，以免误伤 1 字节字段
            
            # 总是包含 0 和 len(msg)
            current_bounds.add(0)
            current_bounds.add(len(msg))
            
            refined_segmentations.append(sorted(list(current_bounds)))
            
        return refined_segmentations