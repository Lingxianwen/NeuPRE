import logging
from typing import List
try:
    from netzob.all import RawMessage, Symbol, Format
    NETZOB_AVAILABLE = True
except ImportError:
    NETZOB_AVAILABLE = False

class DynPRESegmenter:
    """直接调用 Netzob 的对齐算法，模拟 DynPRE 的核心静态分析能力"""
    def __init__(self):
        if not NETZOB_AVAILABLE:
            logging.warning("Netzob not installed. DynPRE will fail.")

    def segment_messages(self, messages: List[bytes]) -> List[List[int]]:
        if not NETZOB_AVAILABLE or not messages:
            return [[0, len(m)] for m in messages]

        try:
            # 1. 转换为 Netzob 对象
            netzob_msgs = [RawMessage(m) for m in messages]
            symbol = Symbol(messages=netzob_msgs)

            # 2. 核心算法：基于对齐的分割 (这是 DynPRE 的静态分析核心)
            Format.splitAligned(symbol, doInternalSlick=False)

            # 3. 提取边界
            segmentations = []
            msg_cells = symbol.getMessageCells()
            
            for msg in symbol.messages:
                boundaries = [0]
                current = 0
                if msg in msg_cells:
                    for field in msg_cells[msg]:
                        # field 是 bytes 或类似对象
                        length = len(field) // 2 if isinstance(field, str) else len(field)
                        if length > 0:
                            current += length
                            boundaries.append(current)
                
                # 修正结尾
                real_len = len(msg.data)
                if boundaries[-1] != real_len:
                    if boundaries[-1] > real_len:
                        boundaries = [b for b in boundaries if b < real_len]
                    boundaries.append(real_len)
                
                segmentations.append(boundaries)
            
            return segmentations
        except Exception as e:
            logging.error(f"Netzob analysis failed: {e}")
            return [[0, len(m)] for m in messages]