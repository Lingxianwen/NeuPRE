"""
NeuPRE Core Algorithms

包含:
- Feature Extraction
- NeuPRE Segmentation
- DYNPRE Perfection Metrics
"""

import logging
import numpy as np
import math
from typing import List, Tuple, Dict
from collections import Counter

from modules.format_learner import InformationBottleneckFormatLearner
from modules.hmm_segmenter import HMMSegmenter
from modules.consensus_refiner import StatisticalConsensusRefiner
from utils.dynpre_segmenter import DynPRESegmenter


# ==================== Feature Extraction ====================

def extract_alignment_features(messages: List[bytes]) -> Tuple[List[float], int]:
    """
    对齐特征提取 + Safe Zone 估计
    
    优化:
    - 增加样本数到 20
    - 降低强边界阈值到 0.4
    - 增加 Safe Zone 余量
    """
    if not messages:
        return [], 0
    
    segmenter = DynPRESegmenter()
    sample_msgs = messages[:20]  # ✅ 增加样本
    segmentations = segmenter.segment_messages(sample_msgs)
    
    if not segmentations:
        return [0.0] * 512, 0

    max_len = 512
    boundary_counts = np.zeros(max_len + 1)
    
    for boundaries in segmentations:
        for b in boundaries:
            if b < max_len:
                boundary_counts[b] += 1
                
    alignment_scores = boundary_counts / len(segmentations)
    
    # 查找最后一个强边界
    last_strong_boundary = 0
    for i in range(len(alignment_scores) - 1, 0, -1):
        if alignment_scores[i] > 0.4:  # ✅ 降低阈值
            last_strong_boundary = i
            break
            
    if last_strong_boundary == 0:
        last_strong_boundary = 32
    else:
        last_strong_boundary += 8  # ✅ 增加余量
        
    logging.info(f"Safe Zone: 0-{last_strong_boundary} bytes")
    return alignment_scores.tolist(), last_strong_boundary


def extract_statistical_features(messages: List[bytes]) -> Tuple[List[float], List[float]]:
    """统计特征提取: 熵差 + 常量切换"""
    if not messages:
        return [], []
    
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
        entropy = 0.0
        total = len(column_bytes)
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        norm_entropy = entropy / 8.0
        entropy_profile.append(norm_entropy)
        
        top_ratio = counts.most_common(1)[0][1] / total
        is_constant.append(1.0 if (norm_entropy < 0.05 or top_ratio > 0.95) else 0.0)
        
    # 熵差
    entropy_diff = [0.0] * len(entropy_profile)
    for i in range(1, len(entropy_profile)):
        diff = abs(entropy_profile[i] - entropy_profile[i-1])
        entropy_diff[i] = diff if diff > 0.1 else 0.0

    # 常量切换点
    const_switch = [0.0] * len(is_constant)
    for i in range(1, len(is_constant)):
        if is_constant[i] != is_constant[i-1]:
            const_switch[i] = 1.0
        
    return entropy_diff, const_switch


# ==================== NeuPRE Segmentation ====================

def simulate_neupre_segmentation(messages: List[bytes]) -> List[List[int]]:
    """
    NeuPRE 完整流程
    
    Steps:
    1. Feature Extraction (对齐 + 统计)
    2. Neural Training (MLM, epochs=30, beta=0.01)
    3. HMM Decoding (Safe Zone 策略)
    4. Consensus Refinement (min_support=0.5)
    """
    # Step 1
    logging.info("Step 1: Feature Extraction...")
    alignment_scores, safe_zone_limit = extract_alignment_features(messages)
    entropy_scores, constant_scores = extract_statistical_features(messages)
    
    # Step 2
    logging.info("Step 2: Neural Training...")
    learner = InformationBottleneckFormatLearner(
        d_model=256, nhead=8, num_layers=4, beta=0.01
    )
    learner.train(messages, messages, epochs=30, batch_size=32)

    # Step 3
    logging.info(f"Step 3: HMM Decoding (Safe Zone = {safe_zone_limit})...")
    hmm = HMMSegmenter()
    hmm.trans_prob = np.log(np.array([[0.6, 0.4], [0.9, 0.1]]) + 1e-10)

    raw_segmentations = []
    
    for msg in messages:
        neural_scores = learner.get_boundary_probs(msg)
        final_scores = []
        max_idx = min(len(msg), 512)
        
        for i in range(max_idx):
            s_align = alignment_scores[i] if i < len(alignment_scores) else 0.0
            s_stat = max(entropy_scores[i], constant_scores[i]) if i < len(entropy_scores) else 0.0
            s_neural = neural_scores[i] if i < len(neural_scores) else 0.5
            
            # Safe Zone 策略
            if i <= safe_zone_limit:
                # 头部: 信任特征
                f_score = max(s_align * 1.0, s_stat * 0.9, s_neural * 0.7)
            else:
                # Payload: 强抑制
                f_score = 0.0
                if s_neural > 0.98:
                    f_score = s_neural * 0.5
            
            final_scores.append(min(1.0, f_score))
            
        boundaries = hmm.segment(final_scores)
        boundaries = sorted(list(set([0] + boundaries + [len(msg)])))
        raw_segmentations.append(boundaries)

    # Step 4
    logging.info("Step 4: Consensus Refinement...")
    refiner = StatisticalConsensusRefiner(min_support=0.5)
    refined_segmentations = refiner.refine(messages, raw_segmentations)

    return refined_segmentations


# ==================== DYNPRE Perfection Metrics ====================

def compute_perfection_single(pred: List[int], gt: List[int]) -> Tuple[float, float]:
    """
    计算单条消息的 DYNPRE Perfection 指标
    
    Field Types:
    - Accurate: 起止都是GT边界, 内部无额外边界
    - Cover: 起止都是GT边界, 内部有额外边界
    - In: 完全在某个GT字段内部
    
    Returns:
        (correctness, perfection)
        - Correctness = (cover + in) / inferred_fields
        - Perfection = accurate / gt_fields
    """
    if not gt or len(gt) < 2 or not pred or len(pred) < 2:
        return 0.0, 0.0
    
    pkt_len = gt[-1]
    pos = [0] * (pkt_len + 1)
    for index in gt:
        if index <= pkt_len:
            pos[index] = 1
    
    cover_num = 0
    in_num = 0
    accurate_num = 0
    
    for i in range(len(pred) - 1):
        start = pred[i]
        end = pred[i + 1]
        
        if start >= len(pos) or end > len(pos):
            continue
        
        both_boundaries = (pos[start] == 1) and (pos[end] == 1)
        has_internal = sum(pos[start + 1:end]) > 0 if start + 1 < end else False
        
        if both_boundaries:
            if not has_internal:
                accurate_num += 1
            else:
                cover_num += 1
        elif not has_internal:
            in_num += 1
    
    inferred_fields = len(pred) - 1
    gt_fields = len(gt) - 1
    
    correctness = (cover_num + in_num) / inferred_fields if inferred_fields > 0 else 0.0
    perfection = accurate_num / gt_fields if gt_fields > 0 else 0.0
    
    return correctness, perfection


def compute_metrics(predicted: List[List[int]], 
                   ground_truth: List[List[int]]) -> Dict[str, float]:
    """
    计算完整指标集
    
    Includes:
    - Boundary-level: Accuracy, Precision, Recall, F1
    - Field-level (DYNPRE): Correctness, Perfection
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_boundaries_gt = 0
    total_correctness = 0.0
    total_perfection = 0.0
    
    for pred, gt in zip(predicted, ground_truth):
        pred_set = set(pred)
        gt_set = set(gt)
        
        # Boundary-level
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_boundaries_gt += len(gt_set)
        
        # Field-level
        correctness, perfection = compute_perfection_single(pred, gt)
        total_correctness += correctness
        total_perfection += perfection
    
    # Boundary metrics
    accuracy = total_tp / total_boundaries_gt if total_boundaries_gt > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Field metrics
    num_messages = len(predicted)
    avg_correctness = total_correctness / num_messages if num_messages > 0 else 0
    avg_perfection = total_perfection / num_messages if num_messages > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correctness': avg_correctness,
        'perfect_match': avg_perfection
    }