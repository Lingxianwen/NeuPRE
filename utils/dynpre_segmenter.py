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
            # 1. 转换数据格式
            netzob_msgs = [RawMessage(m) for m in messages]
            symbol = Symbol(messages=netzob_msgs)

            # 2. 执行核心对齐分割算法
            # doInternalSlick=False 关闭极其耗时的聚类，只做基础对齐
            Format.splitAligned(symbol, doInternalSlick=False)

            # 3. 提取结果
            segmentations = []
            msg_cells = symbol.getMessageCells()
            
            for msg in symbol.messages:
                boundaries = [0]
                current_pos = 0
                if msg in msg_cells:
                    for field in msg_cells[msg]:
                        # field 可能是 bytes 或 string
                        length = len(field) // 2 if isinstance(field, str) else len(field)
                        if length > 0:
                            current_pos += length
                            boundaries.append(current_pos)
                
                # 修正结尾
                real_len = len(msg.data)
                # 移除越界边界，补全结尾边界
                boundaries = [b for b in boundaries if b < real_len]
                boundaries.append(real_len)
                
                segmentations.append(sorted(list(set(boundaries))))
            
            return segmentations
            
        except Exception as e:
            logging.error(f"Netzob static analysis failed: {e}")
            # 失败回退
            return [[0, len(m)] for m in messages]