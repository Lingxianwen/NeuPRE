import logging
from typing import List

# 尝试导入 netzob，它是 DynPRE 的核心依赖
try:
    from netzob.all import RawMessage, Symbol, Format
    NETZOB_AVAILABLE = True
except ImportError:
    NETZOB_AVAILABLE = False

class DynPRESegmenter:
    """
    直接调用 Netzob 的静态对齐算法 (Needleman-Wunsch)。
    这模拟了 DynPRE 在离线模式下的核心能力，且不会报错。
    """
    def __init__(self):
        if not NETZOB_AVAILABLE:
            logging.warning("Netzob not installed. DynPRE segmentation will be empty.")

    def segment_messages(self, messages: List[bytes]) -> List[List[int]]:
        if not NETZOB_AVAILABLE or not messages:
            return [[0, len(m)] for m in messages]

        try:
            # [终极优化 1] 极度截断：只看前 64 字节
            # SMB2 Header 长度固定且较短，64字节足够覆盖关键结构
            MAX_ALIGN_LEN = 64
            truncated_msgs = [m[:MAX_ALIGN_LEN] for m in messages]
            
            # [终极优化 2] 极简采样：只用前 12 条消息学习模板
            # 12条足够发现规律，且计算量是 50条 的 1/16
            analysis_sample_size = 12
            analysis_set = truncated_msgs[:analysis_sample_size]
            
            logging.info(f"Netzob fast-aligning {len(analysis_set)} msgs (len limit={MAX_ALIGN_LEN})...")
            
            netzob_msgs = [RawMessage(m) for m in analysis_set]
            symbol = Symbol(messages=netzob_msgs)

            # 3. 核心算法
            Format.splitAligned(symbol, doInternalSlick=False)

            # 4. 提取通用边界模板
            boundary_counts = {}
            for msg in symbol.messages:
                current = 0
                if msg in symbol.getMessageCells():
                    for field in symbol.getMessageCells()[msg]:
                        l = len(field) // 2 if isinstance(field, str) else len(field)
                        if l > 0:
                            current += l
                            boundary_counts[current] = boundary_counts.get(current, 0) + 1
            
            # 筛选高频边界 (出现在 50% 以上的样本中)
            common_boundaries = [b for b, c in boundary_counts.items() 
                               if c >= len(analysis_set) * 0.5]
            common_boundaries = sorted(list(set([0] + common_boundaries)))
            
            logging.info(f"Learned DynPRE signature: {common_boundaries}")

            # 5. 应用到所有消息
            final_segmentations = []
            for m in messages:
                m_len = len(m)
                # 过滤掉超出长度的，补全结尾
                bounds = [b for b in common_boundaries if b < m_len]
                bounds.append(m_len)
                final_segmentations.append(sorted(list(set(bounds))))
                
            return final_segmentations

        except Exception as e:
            logging.error(f"Netzob analysis failed or timed out: {e}")
            # 失败兜底：不切分
            return [[0, len(m)] for m in messages]