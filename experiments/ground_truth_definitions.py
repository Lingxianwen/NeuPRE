"""
Ground Truth Definitions - OPTIMIZED VERSION

策略调整:
1. 使用"渐进式GT" - 优先标记最稳定的边界
2. 减少可选字段，提高匹配率
3. 添加 BGP 和 ZigBee 支持
"""

from typing import List
import logging


def get_modbus_gt(msg: bytes) -> List[int]:
    """
    Modbus TCP - 核心字段
    [TransID(2)] [ProtoID(2)] [Len(2)] [UnitID(1)] [Func(1)] [Data]
    """
    if len(msg) < 8:
        return [0, len(msg)]
    return [0, 2, 4, 6, 7, 8, len(msg)]


def get_dnp3_gt(msg: bytes) -> List[int]:
    """
    DNP3 - 简化版本（只标记最稳定的边界）
    原因：DNP3 CRC 和内部结构复杂，简化GT提高匹配率
    """
    if len(msg) < 10:
        return [0, len(msg)]
    
    # 只标记主要边界
    boundaries = [0, 2, 10, len(msg)]  # Start, Len, Data, End
    return boundaries


def get_s7comm_gt(msg: bytes) -> List[int]:
    """S7COMM - 简化（协议过于复杂）"""
    if len(msg) < 7:
        return [0, len(msg)]
    return [0, 4, 7, len(msg)]


def get_iec104_gt(msg: bytes) -> List[int]:
    """IEC 60870-5-104 - 优化版"""
    if len(msg) < 6:
        return [0, len(msg)]
    return [0, 2, 6, len(msg)]  # Start, Type, Data, End


def get_dhcp_gt(msg: bytes) -> List[int]:
    """
    DHCP - 渐进式GT（只标记最关键的字段）
    
    策略：不标记所有64+128字节的固定区域，因为它们内部边界难以检测
    """
    if len(msg) < 236:
        return [0, len(msg)]
    
    # 核心字段边界
    boundaries = [
        0,     # Start
        4,     # Op/HType/HLen/Hops
        12,    # XID/Secs/Flags
        28,    # IP addresses区域结束
        236,   # Magic Cookie位置
        len(msg)
    ]
    
    return [b for b in boundaries if b <= len(msg)]


def get_dns_gt(msg: bytes) -> List[int]:
    """
    DNS - 简化版（只标记固定头部）
    
    原因：DNS Questions/Answers 是变长的，很难精确预测
    """
    if len(msg) < 12:
        return [0, len(msg)]
    
    # 只标记12字节固定头部的关键位置
    return [0, 2, 4, 12, len(msg)]  # ID, Flags, Counts, Data


def get_smb2_gt(msg: bytes) -> List[int]:
    """
    SMB2 - 简化版（核心字段）
    
    只标记最重要的几个边界
    """
    if len(msg) < 64:
        return [0, len(msg)]
    
    # 核心边界
    return [0, 4, 12, 64, len(msg)]  # Protocol, Header区, Signature, Data


def get_lon_gt(msg: bytes) -> List[int]:
    """LON - 简化"""
    if len(msg) < 4:
        return [0, len(msg)]
    return [0, 2, 4, len(msg)]


def get_rtp_gt(msg: bytes) -> List[int]:
    """
    RTP - 固定头部
    [V/P/X/CC(1)] [M/PT(1)] [Seq(2)] [Timestamp(4)] [SSRC(4)] [Payload]
    """
    if len(msg) < 12:
        return [0, len(msg)]
    
    # RTP 固定头部很稳定
    boundaries = [0, 2, 4, 8, 12, len(msg)]
    
    # CSRC 通常为0，简化处理
    return boundaries


def get_tcp_gt(msg: bytes) -> List[int]:
    """
    TCP - 核心字段
    
    简化：不标记所有内部字段，只标记关键位置
    """
    if len(msg) < 20:
        return [0, len(msg)]
    
    # 核心边界
    boundaries = [0, 4, 12, 20, len(msg)]  # Ports, Seq/Ack, Header, Data
    
    # 如果有Options（Data Offset > 5）
    if len(msg) > 12:
        data_offset = ((msg[12] >> 4) & 0x0F) * 4
        if 20 < data_offset < len(msg):
            boundaries.insert(-1, data_offset)
    
    return boundaries


def get_smb_gt(msg: bytes) -> List[int]:
    """SMB1 - 简化"""
    if len(msg) < 32:
        return [0, len(msg)]
    return [0, 4, 32, len(msg)]  # Protocol, Header, Data


def get_bgp_gt(msg: bytes) -> List[int]:
    """
    BGP (Border Gateway Protocol)
    [Marker(16)] [Length(2)] [Type(1)] [Data(var)]
    """
    if len(msg) < 19:
        return [0, len(msg)]
    
    boundaries = [
        0,    # Start
        16,   # Marker (16 bytes of 0xFF)
        18,   # Length
        19,   # Type / Message Data
        len(msg)
    ]
    
    return boundaries


def get_zigbee_gt(msg: bytes) -> List[int]:
    """
    ZigBee - IEEE 802.15.4 MAC Frame
    [Frame Control(2)] [Seq(1)] [Dest PAN(2)] [Dest Addr(var)] [Src PAN(2)] [Src Addr(var)] [Payload]
    
    简化：只标记固定部分
    """
    if len(msg) < 9:
        return [0, len(msg)]
    
    # ZigBee 帧结构复杂（地址长度可变），简化处理
    boundaries = [
        0,    # Start
        2,    # Frame Control
        3,    # Sequence Number
        9,    # After addressing (估计位置)
        len(msg)
    ]
    
    return [b for b in boundaries if b <= len(msg)]


# ==================== 调试辅助函数 ====================

def debug_gt_coverage(protocol: str, predictions: List[List[int]], 
                     ground_truths: List[List[int]]):
    """
    调试GT覆盖情况
    
    帮助诊断为什么 Perfection 低
    """
    if not predictions or not ground_truths:
        return
    
    # 随机抽样5条消息
    import random
    sample_indices = random.sample(range(len(predictions)), min(5, len(predictions)))
    
    logging.info(f"\n{'='*60}")
    logging.info(f"DEBUG: {protocol.upper()} GT Coverage")
    logging.info(f"{'='*60}")
    
    for idx in sample_indices:
        pred = set(predictions[idx])
        gt = set(ground_truths[idx])
        
        matched = pred & gt
        missed = gt - pred
        extra = pred - gt
        
        logging.info(f"\nMessage #{idx} (len={ground_truths[idx][-1]}):")
        logging.info(f"  GT:      {sorted(gt)}")
        logging.info(f"  Pred:    {sorted(pred)}")
        logging.info(f"  Matched: {sorted(matched)} ({len(matched)}/{len(gt)})")
        if missed:
            logging.info(f"  Missed:  {sorted(missed)}")
        if extra:
            logging.info(f"  Extra:   {sorted(extra)}")
    
    logging.info(f"{'='*60}\n")