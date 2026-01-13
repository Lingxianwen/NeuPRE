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

def extract_alignment_features(messages: List[bytes]) -> Tuple[List[float], int, dict]:
    """
    IMPROVED: Protocol-adaptive Safe Zone with confidence scoring
    
    Returns:
        - alignment_scores: [0-1] score for each position
        - safe_zone_limit: Estimated header end position
        - metadata: Additional analysis info
    """
    from utils.dynpre_segmenter import DynPRESegmenter
    
    if not messages: 
        return [], 0, {}
    
    # Use more samples for better alignment
    sample_size = min(len(messages), 50)
    segmenter = DynPRESegmenter()
    segmentations = segmenter.segment_messages(messages[:sample_size])
    
    if not segmentations:
        return [0.0] * 512, 32, {}
    
    max_len = 512
    boundary_counts = np.zeros(max_len + 1)
    
    # Collect all boundaries
    for boundaries in segmentations:
        for b in boundaries:
            if b < max_len:
                boundary_counts[b] += 1
    
    # Normalize to [0, 1]
    alignment_scores = boundary_counts / len(segmentations)
    
    # ADAPTIVE Safe Zone Detection
    # Strategy: Find the "cliff" where alignment drops sharply
    
    strong_boundaries = []
    for i in range(1, len(alignment_scores)):
        if alignment_scores[i] > 0.3:  # 30% consensus
            strong_boundaries.append(i)
    
    if not strong_boundaries:
        safe_zone_limit = 32  # Default fallback
    else:
        # Find the last cluster of strong boundaries
        gaps = np.diff(strong_boundaries)
        
        # If gap > 16 bytes, consider it end of header
        large_gap_idx = np.where(gaps > 16)[0]
        
        if len(large_gap_idx) > 0:
            # Safe zone ends at first large gap
            last_header_boundary = strong_boundaries[large_gap_idx[0]]
            safe_zone_limit = last_header_boundary + 4
        else:
            # No large gap found, use last strong boundary
            safe_zone_limit = strong_boundaries[-1] + 4
    
    # Cap at reasonable limits
    safe_zone_limit = min(safe_zone_limit, 256)
    safe_zone_limit = max(safe_zone_limit, 8)  # At least 8 bytes
    
    metadata = {
        'num_strong_boundaries': len(strong_boundaries),
        'max_alignment_score': float(np.max(alignment_scores)),
        'strong_boundaries': strong_boundaries[:10]  # First 10
    }
    
    logging.info(f"Adaptive Safe Zone: {safe_zone_limit} bytes "
                f"({len(strong_boundaries)} strong boundaries)")
    
    return alignment_scores.tolist(), safe_zone_limit, metadata


def extract_statistical_features_v2(messages: List[bytes]) -> dict:
    """
    IMPROVED: More robust statistical features
    
    Returns dict with multiple feature types
    """
    if not messages:
        return {}
    
    max_len = max(len(m) for m in messages)
    analyze_len = min(max_len, 512)
    
    # Feature 1: Entropy per position
    entropy_profile = []
    
    # Feature 2: Variance (high variance = likely variable content)
    variance_profile = []
    
    # Feature 3: Constant detection
    is_constant = []
    
    for i in range(analyze_len):
        column_bytes = [m[i] for m in messages if i < len(m)]
        
        if not column_bytes:
            entropy_profile.append(0.0)
            variance_profile.append(0.0)
            is_constant.append(0.0)
            continue
        
        # Entropy
        counts = Counter(column_bytes)
        total = len(column_bytes)
        entropy = 0.0
        
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        norm_entropy = entropy / 8.0
        entropy_profile.append(norm_entropy)
        
        # Variance
        values = np.array(column_bytes, dtype=float)
        variance = float(np.var(values))
        variance_profile.append(variance)
        
        # Constant detection
        top_ratio = counts.most_common(1)[0][1] / total
        is_constant.append(1.0 if top_ratio > 0.95 else 0.0)
    
    # Feature 4: Entropy gradient (sharp changes indicate boundaries)
    entropy_grad = [0.0] + [
        abs(entropy_profile[i] - entropy_profile[i-1])
        for i in range(1, len(entropy_profile))
    ]
    
    return {
        'entropy': entropy_profile,
        'variance': variance_profile,
        'is_constant': is_constant,
        'entropy_gradient': entropy_grad
    }


def simulate_neupre_segmentation(messages: List[bytes], **kwargs) -> List[List[int]]:
    """
    IMPROVED NeuPRE Segmentation Pipeline
    
    Improvements:
    1. Adaptive Safe Zone per protocol
    2. Better feature fusion
    3. Confidence-based boundary selection
    4. Tolerance-aware consensus refinement
    """
    from modules.format_learner import InformationBottleneckFormatLearner
    from modules.hmm_segmenter import HMMSegmenter
    from modules.consensus_refiner import StatisticalConsensusRefiner
    
    # Step 1: Enhanced Feature Extraction
    logging.info("Step 1: Enhanced Protocol Analysis...")
    alignment_scores, safe_zone_limit, align_meta = extract_alignment_features_v2(messages)
    stat_features = extract_statistical_features_v2(messages)
    
    # Step 2: Neural Model Training
    logging.info("Step 2: Training Neural Boundary Detector...")
    learner = InformationBottleneckFormatLearner(
        d_model=256,
        nhead=8,
        num_layers=4,
        beta=0.01  # Proven value
    )
    
    # Adaptive epochs based on dataset size
    epochs = 30 if len(messages) > 500 else 40
    learner.train(messages, messages, epochs=epochs, batch_size=32)
    
    # Step 3: Multi-Source Boundary Scoring
    logging.info("Step 3: Multi-Source Boundary Fusion...")
    raw_segmentations = []
    
    for msg in messages:
        neural_scores = learner.get_boundary_probs(msg)
        msg_len = min(len(msg), 512)
        
        final_scores = []
        
        for i in range(msg_len):
            # Get scores from each source
            s_align = alignment_scores[i] if i < len(alignment_scores) else 0.0
            s_entropy_grad = stat_features['entropy_gradient'][i] if i < len(stat_features['entropy_gradient']) else 0.0
            s_const_switch = 0.0
            if i > 0 and i < len(stat_features['is_constant']):
                if stat_features['is_constant'][i] != stat_features['is_constant'][i-1]:
                    s_const_switch = 1.0
            
            s_neural = neural_scores[i] if i < len(neural_scores) else 0.5
            
            # IMPROVED Fusion Strategy
            if i <= safe_zone_limit:
                # Header region: Trust alignment > neural > statistics
                # Alignment is very reliable in header
                score = (
                    s_align * 0.50 +          # Alignment is king in header
                    s_neural * 0.30 +         # Neural provides refinement
                    s_entropy_grad * 0.10 +   # Statistical hints
                    s_const_switch * 0.10
                )
                
                # Boost if multiple sources agree
                if s_align > 0.7 and s_neural > 0.6:
                    score = min(1.0, score * 1.2)
                    
            else:
                # Payload region: Be selective
                # Only mark boundaries if neural network is very confident
                # OR if there's strong alignment persistence
                
                if s_neural > 0.95:  # Very high neural confidence
                    score = s_neural * 0.8
                elif s_align > 0.5:  # Alignment persists into payload
                    score = s_align * 0.6
                else:
                    score = 0.0  # Suppress
            
            final_scores.append(min(1.0, score))
        
        # Decode with HMM
        hmm = HMMSegmenter()
        boundaries = hmm.segment(final_scores)
        boundaries = sorted(list(set([0] + boundaries + [len(msg)])))
        raw_segmentations.append(boundaries)
    
    # Step 4: IMPROVED Consensus Refinement with Tolerance
    logging.info("Step 4: Tolerance-Aware Consensus Refinement...")
    
    # Lower min_support for protocols with variable structures
    # Use tolerance=1 to allow ±1 byte differences
    refiner = StatisticalConsensusRefiner(
        min_support=0.25,  # 25% consensus needed
        tolerance=1         # Allow 1-byte differences
    )
    
    refined_segmentations = refiner.refine(messages, raw_segmentations)
    
    # Step 5: Post-Processing - Remove Too-Close Boundaries
    logging.info("Step 5: Post-Processing...")
    final_segmentations = []
    
    for boundaries in refined_segmentations:
        # Remove boundaries that are too close (< 1 byte apart)
        cleaned = [0]
        for i in range(1, len(boundaries) - 1):
            if boundaries[i] - cleaned[-1] >= 1:  # At least 1 byte apart
                cleaned.append(boundaries[i])
        cleaned.append(boundaries[-1])  # Always keep end
        
        final_segmentations.append(cleaned)
    
    return final_segmentations


# Update consensus refiner to support tolerance
class ImprovedConsensusRefiner:
    """
    Enhanced version with tolerance matching
    """
    def __init__(self, min_support: float = 0.25, tolerance: int = 1):
        self.min_support = min_support
        self.tolerance = tolerance
    
    def refine(self, messages: List[bytes], raw_boundaries: List[List[int]]) -> List[List[int]]:
        if not raw_boundaries:
            return raw_boundaries
        
        max_len = max(len(m) for m in messages) if messages else 0
        
        # Build voting matrix with tolerance
        boundary_votes = np.zeros(max_len + 1, dtype=int)
        valid_counts = np.zeros(max_len + 1, dtype=int)
        
        for i, msg in enumerate(messages):
            msg_len = len(msg)
            valid_counts[:msg_len+1] += 1
            
            for b in raw_boundaries[i]:
                if b <= max_len:
                    # Vote for this position AND nearby positions (tolerance)
                    for offset in range(-self.tolerance, self.tolerance + 1):
                        pos = b + offset
                        if 0 <= pos <= max_len:
                            boundary_votes[pos] += 1
        
        # Find consensus boundaries
        consensus_boundaries = set()
        consensus_boundaries.add(0)
        
        for idx in range(1, max_len):
            if valid_counts[idx] < (len(messages) * 0.1):
                continue
            
            support = boundary_votes[idx] / valid_counts[idx]
            
            if support >= self.min_support:
                consensus_boundaries.add(idx)
        
        # Merge close boundaries (within tolerance)
        sorted_bounds = sorted(list(consensus_boundaries))
        merged = [sorted_bounds[0]]
        
        for b in sorted_bounds[1:]:
            if b - merged[-1] > self.tolerance:
                merged.append(b)
        
        logging.info(f"Consensus: {len(merged)} boundaries (from {len(sorted_bounds)} before merge)")
        
        # Apply to each message
        refined_results = []
        for msg in messages:
            msg_len = len(msg)
            final_bounds = {0, msg_len}
            
            for b in merged:
                if b < msg_len:
                    final_bounds.add(b)
            
            refined_results.append(sorted(list(final_bounds)))
        
        return refined_results


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