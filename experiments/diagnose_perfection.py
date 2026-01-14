"""
快速诊断工具 - 分析为什么某些协议 Perfection 低

Usage:
  python diagnose_perfection.py dhcp
  python diagnose_perfection.py dns --samples 10
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from ground_truth_definitions import *
from utils.pcap_loader import PCAPDataLoader

# 协议到PCAP路径的映射
PCAP_PATHS = {
    'dhcp': 'in-dhcp-pcaps/BinInf_dhcp_1000.pcap',
    'dns': 'in-dns-pcaps/SMIA_DNS_1000.pcap',
    'modbus': 'in-modbus-pcaps/libmodbus-bandwidth_server-rand_client.pcap',
    'dnp3': 'in-dnp3-pcaps/BinInf_dnp3_1000.pcap',
    'tcp': 'in-tcp-pcaps/SMIA_TCP_part_1000.pcap',
    's7comm': 'in-s7comm-pcaps/s7comm.pcap',
    'smb2': 'in-smb2-pcaps/samba.pcap',
}


def visualize_message(msg: bytes, gt: list, pred: list):
    """可视化单条消息的分割结果"""
    print(f"\n{'='*80}")
    print(f"Message Length: {len(msg)} bytes")
    print(f"{'='*80}")
    
    # GT 字段
    print("\n📋 Ground Truth Fields:")
    for i in range(len(gt) - 1):
        start, end = gt[i], gt[i+1]
        field_bytes = msg[start:end]
        print(f"  Field {i}: [{start:3d}:{end:3d}] ({end-start:2d} bytes) | {field_bytes[:8].hex()}...")
    
    # Pred 字段（模拟）
    print("\n🔮 Would-be Predicted Fields (example):")
    print("   [假设NeuPRE找到的边界]")
    for i in range(len(pred) - 1):
        start, end = pred[i], pred[i+1]
        if end <= len(msg):
            field_bytes = msg[start:end]
            # 检查是否匹配GT
            match = "✅" if (start in gt and end in gt) else "❌"
            print(f"  {match} Field {i}: [{start:3d}:{end:3d}] ({end-start:2d} bytes) | {field_bytes[:8].hex()}...")
    
    # 对比
    print("\n📊 Comparison:")
    gt_set = set(zip(gt[:-1], gt[1:]))
    pred_set = set(zip(pred[:-1], pred[1:]))
    
    accurate = gt_set & pred_set
    missed = gt_set - pred_set
    extra = pred_set - gt_set
    
    print(f"  ✅ Accurate fields: {len(accurate)}/{len(gt_set)} ({len(accurate)/len(gt_set)*100:.1f}%)")
    if accurate:
        print(f"     {list(accurate)}")
    
    if missed:
        print(f"  ❌ Missed GT fields: {len(missed)}")
        print(f"     {list(missed)}")
    
    if extra:
        print(f"  ⚠️  Extra predicted fields: {len(extra)}")
        print(f"     {list(extra)}")


def analyze_protocol(protocol: str, num_samples: int = 5):
    """分析协议的GT定义"""
    print(f"\n{'='*80}")
    print(f"Analyzing {protocol.upper()}")
    print(f"{'='*80}")
    
    # 加载数据
    if protocol not in PCAP_PATHS:
        print(f"❌ Unknown protocol: {protocol}")
        print(f"Available: {list(PCAP_PATHS.keys())}")
        return
    
    loader = PCAPDataLoader(data_dir='../data')
    messages = loader.load_messages(PCAP_PATHS[protocol], max_messages=100)
    
    if not messages:
        print(f"❌ No messages loaded")
        return
    
    print(f"✅ Loaded {len(messages)} messages")
    
    # 生成GT
    gt_func = globals()[f'get_{protocol}_gt']
    
    # 统计信息
    field_counts = []
    field_sizes = []
    
    for msg in messages:
        gt = gt_func(msg)
        field_counts.append(len(gt) - 1)
        for i in range(len(gt) - 1):
            field_sizes.append(gt[i+1] - gt[i])
    
    print(f"\n📊 Statistics:")
    print(f"  Avg fields per message: {np.mean(field_counts):.2f} ± {np.std(field_counts):.2f}")
    print(f"  Avg field size: {np.mean(field_sizes):.2f} bytes")
    print(f"  Min/Max field size: {min(field_sizes)}/{max(field_sizes)} bytes")
    
    # 显示样本
    import random
    samples = random.sample(range(len(messages)), min(num_samples, len(messages)))
    
    for idx in samples:
        msg = messages[idx]
        gt = gt_func(msg)
        
        # 模拟一个简单的预测（用于演示）
        # 实际预测需要运行完整的NeuPRE
        pred = [0, len(msg)]  # 最简单的预测：只有开头和结尾
        
        visualize_message(msg, gt, pred)
    
    # 建议
    print(f"\n💡 Recommendations:")
    avg_fields = np.mean(field_counts)
    
    if avg_fields > 10:
        print(f"  ⚠️  GT定义了 {avg_fields:.0f} 个字段，可能太详细")
        print(f"     建议：简化GT，只标记最稳定的边界")
    elif avg_fields < 3:
        print(f"  ⚠️  GT只有 {avg_fields:.0f} 个字段，可能太简单")
        print(f"     建议：增加关键边界")
    else:
        print(f"  ✅ GT定义合理 ({avg_fields:.0f} 个字段)")
    
    if min(field_sizes) < 2:
        print(f"  ⚠️  存在 1 字节的字段，检测难度高")
    
    if max(field_sizes) > 100:
        print(f"  ℹ️  存在大字段 ({max(field_sizes)} 字节)")
        print(f"     大字段内部边界难以检测")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnose protocol Perfection issues')
    parser.add_argument('protocol', help='Protocol to analyze')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to show')
    args = parser.parse_args()
    
    import numpy as np
    analyze_protocol(args.protocol, args.samples)