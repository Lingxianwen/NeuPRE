"""
Experiment 2: Field Boundary Accuracy (Safe-Zone Ensemble)
Logic:
1. Use DynPRE alignment to estimate the "Header Size" (e.g., last aligned boundary).
2. Force HMM to be conservative outside this "Safe Zone".
3. This eliminates false positives in the variable-length payload.
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

# ================= Feature Extraction =================

def run_real_dynpre_static(messages: List[bytes]) -> List[List[int]]:
    logging.info("Running Baseline: Static DynPRE (Netzob alignment)...")
    segmenter = DynPRESegmenter()
    return segmenter.segment_messages(messages)

def extract_alignment_features(messages: List[bytes]) -> Tuple[List[float], int]:
    """
    提取对齐特征，并估计 Header 的大致长度（Safe Zone）。
    """
    if not messages: return [], 0
    
    segmenter = DynPRESegmenter()
    sample_msgs = messages[:12]
    segmentations = segmenter.segment_messages(sample_msgs)
    
    if not segmentations: return [0.0] * 512, 0

    max_len = 512
    boundary_counts = np.zeros(max_len + 1)
    last_strong_boundary = 0
    
    for boundaries in segmentations:
        for b in boundaries:
            if b < max_len:
                boundary_counts[b] += 1
                
    alignment_scores = boundary_counts / len(segmentations)
    
    # 寻找最后一个强边界 (Probability > 0.5)
    # 这通常标志着 Header 的结束
    for i in range(len(alignment_scores) - 1, 0, -1):
        if alignment_scores[i] > 0.5:
            last_strong_boundary = i
            break
            
    # 如果没找到（比如 SMB2），保守估计为 32
    if last_strong_boundary == 0:
        last_strong_boundary = 32
    else:
        # 给点余量
        last_strong_boundary += 4
        
    logging.info(f"Estimated Header Safe Zone: 0 - {last_strong_boundary} bytes")
    return alignment_scores.tolist(), last_strong_boundary

def extract_statistical_features(messages: List[bytes]) -> Tuple[List[float], List[float]]:
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
        entropy = 0.0
        total = len(column_bytes)
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        
        norm_entropy = entropy / 8.0
        entropy_profile.append(norm_entropy)
        
        top_ratio = counts.most_common(1)[0][1] / total
        is_constant.append(1.0 if (norm_entropy < 0.05 or top_ratio > 0.95) else 0.0)
        
    entropy_diff = [0.0] * len(entropy_profile)
    for i in range(1, len(entropy_profile)):
        diff = abs(entropy_profile[i] - entropy_profile[i-1])
        entropy_diff[i] = diff if diff > 0.1 else 0.0

    const_switch = [0.0] * len(is_constant)
    for i in range(1, len(is_constant)):
        if is_constant[i] != is_constant[i-1]:
            const_switch[i] = 1.0
        
    return entropy_diff, const_switch

# ================= Core Brain =================

def simulate_neupre_segmentation(messages: List[bytes], use_supervised=False, **kwargs) -> List[List[int]]:
    """
    NeuPRE: Safe-Zone Ensemble
    """
    # 1. 提取特征 & 确定安全区
    logging.info("Step 1: Analyzing Protocol Structure...")
    alignment_scores, safe_zone_limit = extract_alignment_features(messages)
    entropy_scores, constant_scores = extract_statistical_features(messages)
    
    # 2. 训练
    logging.info("Step 2: Training Neural Model...")
    learner = InformationBottleneckFormatLearner(d_model=256, nhead=8, num_layers=4, beta=0.01)
    learner.train(messages, messages, epochs=30, batch_size=32)

    # 3. 推理
    logging.info(f"Step 3: Decoding with Safe Zone Limit = {safe_zone_limit}...")
    hmm = HMMSegmenter()
    # 默认 HMM 参数
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
            
            # --- [核心逻辑] 安全区控制 ---
            if i <= safe_zone_limit:
                # 在安全区内：火力全开
                # 信任对齐 > 统计 > 神经
                f_score = max(s_align * 1.0, s_stat * 0.9, s_neural * 0.7)
            else:
                # 超出安全区 (Payload)：进入静默模式
                # 强制压低所有分数，除非神经信号极其强烈 (>0.95)
                # 这样可以防止 Payload 里的过分割
                f_score = 0.0 # 激进策略：Payload 不切！
                if s_neural > 0.98: # 除非 BERT 非常确信
                    f_score = s_neural * 0.5
            
            final_scores.append(min(1.0, f_score))
            
        boundaries = hmm.segment(final_scores)
        boundaries = sorted(list(set([0] + boundaries + [len(msg)])))
        raw_segmentations.append(boundaries)

    # 4. Refine
    logging.info("Step 4: Refinement...")
    refiner = StatisticalConsensusRefiner(min_support=0.6)
    refined_segmentations = refiner.refine(messages, raw_segmentations)

    return refined_segmentations

# ================= Runner =================

def run_experiment2(num_samples: int = 1000,
                   output_dir: str = './experiments/results',
                   use_real_data: bool = True,
                   use_dynpre_ground_truth: bool = False):
    setup_logging(level=logging.INFO)
    logging.info("=" * 80)
    logging.info("EXPERIMENT 2: Field Boundary Accuracy (Safe-Zone)")
    logging.info("=" * 80)

    evaluator = NeuPREEvaluator(output_dir=output_dir)
    protocols = {}

    if use_real_data:
        loader = PCAPDataLoader(data_dir='data')
        
        logging.info("Loading DNP3 data...")
        dnp3_msgs = loader.load_messages("in-dnp3-pcaps/BinInf_dnp3_1000.pcap") 
        if dnp3_msgs:
            dnp3_gt = []
            for msg in dnp3_msgs:
                b = [0]
                if len(msg) >= 10: b.append(10)
                b.append(len(msg))
                dnp3_gt.append(sorted(list(set(b))))
            protocols['dnp3'] = (dnp3_msgs, dnp3_gt)
            logging.info(f"Loaded {len(dnp3_msgs)} DNP3 messages")

        try:
            msgs, gt = loader.load_protocol_data('modbus', max_messages=num_samples)
            if msgs: protocols['modbus'] = (msgs, gt)
        except: pass

        # 3. [新增/确认] DHCP
        try:
            logging.info("Loading DHCP data...")
            # 确保你的 PCAPDataLoader 有 load_protocol_data 方法能处理 dhcp
            # 或者使用 loader.load_messages + 自定义 GT
            msgs, gt = loader.load_protocol_data('dhcp', max_messages=num_samples)
            if msgs: 
                protocols['dhcp'] = (msgs, gt)
                logging.info(f"Loaded {len(msgs)} DHCP messages")
        except Exception as e:
            logging.warning(f"Skipping dhcp: {e}")

        # 4. [新增/确认] DNS
        try:
            logging.info("Loading DNS data...")
            msgs, gt = loader.load_protocol_data('dns', max_messages=num_samples)
            if msgs: 
                protocols['dns'] = (msgs, gt)
                logging.info(f"Loaded {len(msgs)} DNS messages")
        except Exception as e:
            logging.warning(f"Skipping dns: {e}")
            
        # 5. [新增/确认] SMB2 (之前会卡死，现在应该好了)
        try:
            logging.info("Loading SMB2 data...")
            msgs, gt = loader.load_protocol_data('smb2', max_messages=num_samples)
            if msgs: 
                protocols['smb2'] = (msgs, gt)
                logging.info(f"Loaded {len(msgs)} SMB2 messages")
        except Exception as e:
            logging.warning(f"Skipping smb2: {e}")

    for protocol_name, (messages, ground_truth) in protocols.items():
        logging.info(f"\n{'='*80}")
        logging.info(f"Testing on {protocol_name} protocol")
        logging.info(f"{'='*80}")

        neupre_boundaries = simulate_neupre_segmentation(messages)
        dynpre_boundaries = run_real_dynpre_static(messages)

        neupre_metrics = evaluator.evaluate_segmentation_accuracy(neupre_boundaries, ground_truth)
        dynpre_metrics = evaluator.evaluate_segmentation_accuracy(dynpre_boundaries, ground_truth)
        
        comparison = evaluator.compare_methods(neupre_metrics, dynpre_metrics, 'segmentation')
        
        evaluator.plot_segmentation_comparison(
            neupre_metrics, dynpre_metrics, 
            filename=f'segmentation_{protocol_name}.png'
        )

        logging.info(f"\n{protocol_name.upper()} Results:")
        logging.info(f"NeuPRE F1: {neupre_metrics.f1_score:.4f}")
        logging.info(f"DYNpre F1: {dynpre_metrics.f1_score:.4f}")
        logging.info(f"Improvement: {comparison['improvement'].get('f1_score', 0):.2f}%")

    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY")
    logging.info("=" * 80)

if __name__ == '__main__':
    run_experiment2(num_samples=1000, output_dir='./experiments/results', use_real_data=True)