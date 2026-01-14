"""
Experiment 2: Field Boundary Accuracy - FIXED VERSION

主要修改：
1. 改进Safe Zone策略：不再完全禁止Payload区域的切分
2. 增加对齐特征的权重
3. 增加训练轮数和改进神经网络架构
4. 改进共识精炼器的支持度阈值
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import math
from collections import Counter
from typing import List, Tuple

from neupre import setup_logging
from utils.evaluator import NeuPREEvaluator
from utils.pcap_loader import PCAPDataLoader
from utils.dynpre_loader import DynPREGroundTruthLoader

# Modules
from modules.format_learner import InformationBottleneckFormatLearner
from modules.hmm_segmenter import HMMSegmenter
from modules.consensus_refiner import StatisticalConsensusRefiner
from utils.dynpre_segmenter import DynPRESegmenter


def get_dnp3_real_gt(msg: bytes) -> List[int]:
    """真实的 DNP3 边界定义: [Start(2)][Len(1)][Ctrl(1)][Dest(2)][Src(2)][CRC(2)][Data]"""
    # 只有当消息足够长时才标记这些字段
    boundaries = [0]
    # DNP3 头部固定字段位置
    candidates = [2, 3, 4, 6, 8, 10]
    for b in candidates:
        if b < len(msg):
            boundaries.append(b)
    boundaries.append(len(msg))
    return sorted(list(set(boundaries)))

def run_real_dynpre_static(messages: List[bytes]) -> List[List[int]]:
    logging.info("Running Baseline: Static DynPRE (Netzob alignment)...")
    segmenter = DynPRESegmenter()
    return segmenter.segment_messages(messages)


# def extract_alignment_features(messages: List[bytes]) -> Tuple[List[float], int]:
#     """改进的对齐特征提取"""
#     if not messages: return [], 0
    
#     segmenter = DynPRESegmenter()
#     sample_msgs = messages[:15]  # 增加样本数
#     segmentations = segmenter.segment_messages(sample_msgs)
    
#     if not segmentations: return [0.0] * 512, 0

#     max_len = 512
#     boundary_counts = np.zeros(max_len + 1)
#     last_strong_boundary = 0
    
#     for boundaries in segmentations:
#         for b in boundaries:
#             if b < max_len:
#                 boundary_counts[b] += 1
                
#     alignment_scores = boundary_counts / len(segmentations)
    
#     # 改进：使用更灵活的Safe Zone估计
#     strong_boundaries = []
#     for i in range(len(alignment_scores)):
#         if alignment_scores[i] > 0.6:  # 提高阈值
#             strong_boundaries.append(i)
    
#     if strong_boundaries:
#         # Safe Zone = 最后一个强边界 + 适度余量
#         last_strong_boundary = max(strong_boundaries) + 8
#     else:
#         last_strong_boundary = 32
        
#     # 限制最大Safe Zone长度
#     last_strong_boundary = min(last_strong_boundary, 64)
        
#     logging.info(f"Estimated Header Safe Zone: 0 - {last_strong_boundary} bytes")
#     return alignment_scores.tolist(), last_strong_boundary


# def extract_statistical_features(messages: List[bytes]) -> Tuple[List[float], List[float]]:
#     """改进的统计特征提取"""
#     if not messages: return [], []
    
#     max_len = max(len(m) for m in messages)
#     analyze_len = min(max_len, 512)
    
#     entropy_profile = []
#     is_constant = []
    
#     for i in range(analyze_len):
#         column_bytes = [m[i] for m in messages if i < len(m)]
#         if not column_bytes:
#             entropy_profile.append(0.0)
#             is_constant.append(0.0)
#             continue
            
#         counts = Counter(column_bytes)
#         entropy = 0.0
#         total = len(column_bytes)
#         for count in counts.values():
#             p = count / total
#             entropy -= p * math.log2(p)
        
#         norm_entropy = entropy / 8.0
#         entropy_profile.append(norm_entropy)
        
#         top_ratio = counts.most_common(1)[0][1] / total
#         is_constant.append(1.0 if (norm_entropy < 0.05 or top_ratio > 0.95) else 0.0)
        
#     entropy_diff = [0.0] * len(entropy_profile)
#     for i in range(1, len(entropy_profile)):
#         diff = abs(entropy_profile[i] - entropy_profile[i-1])
#         entropy_diff[i] = diff if diff > 0.15 else 0.0  # 提高阈值

#     const_switch = [0.0] * len(is_constant)
#     for i in range(1, len(is_constant)):
#         if is_constant[i] != is_constant[i-1]:
#             const_switch[i] = 1.0
        
#     return entropy_diff, const_switch


# def simulate_neupre_segmentation(messages: List[bytes], use_supervised=False, **kwargs) -> List[List[int]]:
#     """
#     改进的NeuPRE分割算法
#     """
#     # 1. 特征提取
#     logging.info("Step 1: Analyzing Protocol Structure...")
#     alignment_scores, safe_zone_limit = extract_alignment_features(messages)
#     entropy_scores, constant_scores = extract_statistical_features(messages)
    
#     # 2. 训练神经模型（增加轮数）
#     logging.info("Step 2: Training Neural Model...")
#     learner = InformationBottleneckFormatLearner(
#         d_model=256, 
#         nhead=8, 
#         num_layers=4, 
#         beta=0.005  # 降低beta以减少过度稀疏化
#     )
#     learner.train(messages, messages, epochs=50, batch_size=32)  # 增加到50轮

#     # 3. 推理（改进的Safe Zone策略）
#     logging.info(f"Step 3: Decoding with IMPROVED Safe Zone Strategy (limit={safe_zone_limit})...")
#     hmm = HMMSegmenter()
#     hmm.trans_prob = np.log(np.array([[0.65, 0.35], [0.85, 0.15]]) + 1e-10)

#     raw_segmentations = []
    
#     for msg in messages:
#         neural_scores = learner.get_boundary_probs(msg)
#         final_scores = []
#         max_idx = min(len(msg), 512)
        
#         for i in range(max_idx):
#             s_align = alignment_scores[i] if i < len(alignment_scores) else 0.0
#             s_stat = max(entropy_scores[i], constant_scores[i]) if i < len(entropy_scores) else 0.0
#             s_neural = neural_scores[i] if i < len(neural_scores) else 0.5
            
#             # 改进的Safe Zone控制
#             if i <= safe_zone_limit:
#                 # Safe Zone内：信任对齐 > 神经网络 > 统计
#                 f_score = max(
#                     s_align * 1.2,      # 提高对齐权重
#                     s_neural * 0.9,
#                     s_stat * 0.8
#                 )
#             else:
#                 # Payload区域：改为适度切分而非完全禁止
#                 # 提高神经网络的阈值，但不完全禁止
#                 if s_neural > 0.95:  # 只有非常确信的边界才保留
#                     f_score = s_neural * 0.7
#                 elif s_align > 0.4:  # 如果对齐特征仍然较强
#                     f_score = s_align * 0.6
#                 else:
#                     f_score = 0.0  # 其他情况才禁止
            
#             final_scores.append(min(1.0, f_score))
            
#         boundaries = hmm.segment(final_scores)
#         boundaries = sorted(list(set([0] + boundaries + [len(msg)])))
#         raw_segmentations.append(boundaries)

#     # 4. 改进的Refinement
#     logging.info("Step 4: Refinement...")
#     refiner = StatisticalConsensusRefiner(min_support=0.5)  # 降低到50%
#     refined_segmentations = refiner.refine(messages, raw_segmentations)

#     return refined_segmentations

def extract_alignment_features(messages: List[bytes]) -> Tuple[List[float], int]:
    """改进的对齐特征提取：移除硬性的 64 字节限制"""
    if not messages: return [], 0
    
    # 增加样本量以获得更准的对齐
    sample_size = min(len(messages), 50)
    segmenter = DynPRESegmenter()
    segmentations = segmenter.segment_messages(messages[:sample_size])
    
    if not segmentations: return [0.0] * 512, 0

    max_len = 512
    boundary_counts = np.zeros(max_len + 1)
    
    for boundaries in segmentations:
        for b in boundaries:
            if b < max_len:
                boundary_counts[b] += 1
                
    alignment_scores = boundary_counts / len(segmentations)
    
    # 动态计算 Safe Zone
    # 寻找对齐分数较高的区域，不设硬性上限，只设一个合理的物理上限 (如 256) 
    strong_indices = [i for i, score in enumerate(alignment_scores) if score > 0.3]
    
    if strong_indices:
        # 延伸到最后一个强特征点之后一点，上限放宽到 256 以容纳 DHCP/SMB
        last_strong_boundary = min(max(strong_indices) + 4, 256)
    else:
        last_strong_boundary = 32
        
    logging.info(f"Estimated Header Safe Zone: 0 - {last_strong_boundary} bytes")
    return alignment_scores.tolist(), last_strong_boundary

def extract_statistical_features(messages: List[bytes]) -> Tuple[List[float], List[float]]:
    """统计特征提取 (保持原逻辑，微调阈值)"""
    if not messages: return [], []
    
    max_len = max(len(m) for m in messages)
    analyze_len = min(max_len, 512)
    
    entropy_profile = []
    is_constant = []
    
    for i in range(analyze_len):
        column_bytes = [m[i] for m in messages if i < len(m)]
        if not column_bytes:
            entropy_profile.append(0.0)
            is_constant.append(0.0)
            continue
            
        counts = Counter(column_bytes)
        total = len(column_bytes)
        
        # 归一化熵
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        entropy_profile.append(entropy / 8.0)
        
        # 常量检测 (放宽到 90% 占比)
        top_ratio = counts.most_common(1)[0][1] / total
        is_constant.append(1.0 if top_ratio > 0.90 else 0.0)
        
    return entropy_profile, is_constant

def simulate_neupre_segmentation(messages: List[bytes], use_supervised=False, **kwargs) -> List[List[int]]:
    """
    改进的 NeuPRE 分割流程
    """
    # 1. 特征提取
    logging.info("Step 1: Analyzing Protocol Structure...")
    alignment_scores, safe_zone_limit = extract_alignment_features(messages)
    entropy_scores, constant_scores = extract_statistical_features(messages)
    
    # 2. 训练神经模型 (保持 30-50 轮)
    logging.info("Step 2: Training Neural Model...")
    learner = InformationBottleneckFormatLearner(
        d_model=256, nhead=8, num_layers=4, beta=0.005
    )
    learner.train(messages, messages, epochs=40, batch_size=32)

    # 3. 推理 (融合逻辑优化)
    logging.info(f"Step 3: Decoding (Safe Zone limit={safe_zone_limit})...")
    hmm = HMMSegmenter()
    hmm.trans_prob = np.log(np.array([[0.7, 0.3], [0.9, 0.1]]) + 1e-10)

    raw_segmentations = []
    
    for msg in messages:
        neural_scores = learner.get_boundary_probs(msg)
        final_scores = []
        max_idx = min(len(msg), 512)
        
        for i in range(max_idx):
            s_align = alignment_scores[i] if i < len(alignment_scores) else 0.0
            s_stat = 1.0 - entropy_scores[i] if i < len(entropy_scores) else 0.0 # 熵越低越可能是边界
            s_neural = neural_scores[i] if i < len(neural_scores) else 0.5
            
            # 融合策略
            if i <= safe_zone_limit:
                # 头部区域：综合考虑，增加了对齐分数的权重
                f_score = (s_align * 0.5) + (s_neural * 0.3) + (s_stat * 0.2)
                # 如果对齐极强，给予额外加成
                if s_align > 0.8: f_score += 0.2
            else:
                # 载荷区域：主要信任神经网络，允许发现深层边界
                f_score = s_neural * 0.8
                # 仅抑制极低置信度
                if f_score < 0.3: f_score = 0.0
            
            final_scores.append(min(1.0, f_score))
            
        boundaries = hmm.segment(final_scores)
        boundaries = sorted(list(set([0] + boundaries + [len(msg)])))
        raw_segmentations.append(boundaries)

    # 4. Refinement (关键修改: 降低阈值)
    logging.info("Step 4: Consensus Refinement...")
    # min_support 降至 0.25 (25%) 以捕获可选字段
    # tolerance 设为 1 允许 1 字节的误差合并
    refiner = StatisticalConsensusRefiner(min_support=0.25, tolerance=1)
    refined_segmentations = refiner.refine(messages, raw_segmentations)

    return refined_segmentations


def run_experiment2(num_samples: int = 1000,
                   output_dir: str = './experiments/results',
                   use_real_data: bool = True,
                   use_dynpre_ground_truth: bool = False):
    """改进的实验2主函数"""
    setup_logging(level=logging.INFO)
    logging.info("=" * 80)
    logging.info("EXPERIMENT 2: Field Boundary Accuracy [FIXED VERSION]")
    logging.info("=" * 80)

    evaluator = NeuPREEvaluator(output_dir=output_dir)
    protocols = {}

    if use_real_data:
        loader = PCAPDataLoader(data_dir='data')
        
        # 1. DNP3
        # logging.info("Loading DNP3 data...")
        # dnp3_msgs = loader.load_messages("in-dnp3-pcaps/BinInf_dnp3_1000.pcap") 
        # if dnp3_msgs:
        #     dnp3_gt = []
        #     for msg in dnp3_msgs:
        #         b = [0]
        #         if len(msg) >= 10: b.append(10)
        #         b.append(len(msg))
        #         dnp3_gt.append(sorted(list(set(b))))
        #     protocols['dnp3'] = (dnp3_msgs, dnp3_gt)
        #     logging.info(f"Loaded {len(dnp3_msgs)} DNP3 messages")
        logging.info("Loading DNP3 data...")
        # 尝试加载数据
        dnp3_msgs = loader.load_messages("in-dnp3-pcaps/BinInf_dnp3_1000.pcap")
        if dnp3_msgs:
            # 使用真实的 GT 函数，而不是硬编码
            dnp3_gt = [get_dnp3_real_gt(m) for m in dnp3_msgs]
            protocols['dnp3'] = (dnp3_msgs, dnp3_gt)
            logging.info(f"Loaded {len(dnp3_msgs)} DNP3 messages with REAL Ground Truth")

        # 2. Modbus
        try:
            msgs, gt = loader.load_protocol_data('modbus', max_messages=num_samples)
            if msgs: protocols['modbus'] = (msgs, gt)
        except: pass

        # 3. DHCP
        try:
            logging.info("Loading DHCP data...")
            msgs, gt = loader.load_protocol_data('dhcp', max_messages=num_samples)
            if msgs: 
                protocols['dhcp'] = (msgs, gt)
                logging.info(f"Loaded {len(msgs)} DHCP messages")
        except Exception as e:
            logging.warning(f"Skipping dhcp: {e}")

        # 4. DNS
        try:
            logging.info("Loading DNS data...")
            msgs, gt = loader.load_protocol_data('dns', max_messages=num_samples)
            if msgs: 
                protocols['dns'] = (msgs, gt)
                logging.info(f"Loaded {len(msgs)} DNS messages")
        except Exception as e:
            logging.warning(f"Skipping dns: {e}")

    # 测试每个协议
    all_comparisons = []
    
    for protocol_name, (messages, ground_truth) in protocols.items():
        logging.info(f"\n{'='*80}")
        logging.info(f"Testing on {protocol_name} protocol")
        logging.info(f"{'='*80}")

        neupre_boundaries = simulate_neupre_segmentation(messages)
        dynpre_boundaries = run_real_dynpre_static(messages)

        neupre_metrics = evaluator.evaluate_segmentation_accuracy(neupre_boundaries, ground_truth)
        dynpre_metrics = evaluator.evaluate_segmentation_accuracy(dynpre_boundaries, ground_truth)
        
        comparison = evaluator.compare_methods(neupre_metrics, dynpre_metrics, 'segmentation')
        all_comparisons.append(comparison)
        
        evaluator.plot_segmentation_comparison(
            neupre_metrics, dynpre_metrics, 
            filename=f'segmentation_{protocol_name}.png'
        )

        logging.info(f"\n{protocol_name.upper()} Results:")
        logging.info(f"NeuPRE F1: {neupre_metrics.f1_score:.4f}")
        logging.info(f"DYNpre F1: {dynpre_metrics.f1_score:.4f}")
        logging.info(f"Improvement: {comparison['improvement'].get('f1_score', 0):.2f}%")

    # 总结
    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY [FIXED VERSION]")
    logging.info("=" * 80)
    
    # 计算平均改进
    avg_improvement = np.mean([
        comp['improvement'].get('f1_score', 0) 
        for comp in all_comparisons
    ])
    logging.info(f"Average F1 improvement across all protocols: {avg_improvement:.2f}%")
    logging.info("=" * 80)


if __name__ == '__main__':
    run_experiment2(
        num_samples=1000, 
        output_dir='./experiments/results', 
        use_real_data=True
    )