"""
Experiment 2: Field Boundary Accuracy

Compares field segmentation accuracy between NeuPRE (IB-based) and DYNpre (heuristic-based).

Protocols tested:
- Real protocols from pcap files (DHCP, DNS, Modbus, SMB2)
- Custom synthetic protocols (for comparison)

Expected result:
- NeuPRE achieves higher F1-Score and Perfect Match Rate
- NeuPRE handles high-entropy fields better than DYNpre
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
from typing import List, Tuple, Dict

from neupre import NeuPRE, setup_logging
from utils.evaluator import NeuPREEvaluator
from utils.pcap_loader import PCAPDataLoader
from utils.dynpre_loader import DynPREGroundTruthLoader
from modules.format_learner import InformationBottleneckFormatLearner

from modules.refiner import GlobalAlignmentRefiner  # 新增
from utils.real_dynpre_wrapper import RealDynPRERunner # 新增


class ProtocolDataset:
    """Synthetic protocol datasets with ground truth"""

    @staticmethod
    def generate_simple_protocol(num_samples: int = 100) -> Tuple[List[bytes], List[List[int]]]:
        """
        Simple protocol: [Header(2)] [Type(1)] [Length(1)] [Payload(variable)] [Checksum(1)]
        """
        messages = []
        ground_truth = []

        for _ in range(num_samples):
            header = b'\xAA\xBB'
            msg_type = bytes([np.random.randint(0, 10)])
            payload_len = np.random.randint(4, 16)
            payload = bytes(np.random.randint(0, 256, payload_len))
            checksum = bytes([sum(payload) % 256])

            message = header + msg_type + bytes([payload_len]) + payload + checksum
            messages.append(message)

            # Ground truth boundaries
            gt = [0, 2, 3, 4, 4 + payload_len, 4 + payload_len + 1]
            ground_truth.append(gt)

        return messages, ground_truth

    @staticmethod
    def generate_high_entropy_protocol(num_samples: int = 100) -> Tuple[List[bytes], List[List[int]]]:
        """
        High-entropy protocol with random-looking fields (simulating encryption)
        [Magic(4)] [Random(8)] [Type(1)] [Random(16)] [Length(2)] [Payload(variable)]
        """
        messages = []
        ground_truth = []

        for _ in range(num_samples):
            magic = b'\xDE\xAD\xBE\xEF'
            random1 = bytes(np.random.randint(0, 256, 8))
            msg_type = bytes([np.random.randint(0, 5)])
            random2 = bytes(np.random.randint(0, 256, 16))
            payload_len = np.random.randint(8, 32)
            length_field = payload_len.to_bytes(2, byteorder='big')
            payload = bytes(np.random.randint(0, 256, payload_len))

            message = magic + random1 + msg_type + random2 + length_field + payload
            messages.append(message)

            # Ground truth
            gt = [0, 4, 12, 13, 29, 31, 31 + payload_len]
            ground_truth.append(gt)

        return messages, ground_truth

    @staticmethod
    def generate_mixed_protocol(num_samples: int = 100) -> Tuple[List[bytes], List[List[int]]]:
        """
        Mixed text/binary protocol
        [BinaryHeader(4)] [TextCommand(variable)] [BinaryLength(2)] [BinaryPayload(variable)]
        """
        messages = []
        ground_truth = []

        commands = [b'GET', b'POST', b'PUT', b'DELETE', b'HEAD']

        for _ in range(num_samples):
            header = b'\x01\x02\x03\x04'
            command = np.random.choice(commands)
            payload_len = np.random.randint(4, 20)
            length_field = payload_len.to_bytes(2, byteorder='big')
            payload = bytes(np.random.randint(0, 256, payload_len))

            message = header + command + length_field + payload
            messages.append(message)

            # Ground truth
            gt = [0, 4, 4 + len(command), 4 + len(command) + 2, 4 + len(command) + 2 + payload_len]
            ground_truth.append(gt)

        return messages, ground_truth


def run_real_dynpre(messages: List[bytes]) -> List[List[int]]:
    # 假设 DynPRE 路径在项目根目录的同级或特定位置，请根据你的实际路径修改
    # 你之前是在 ./DynPRE/DynPRE.py
    dynpre_path = os.path.abspath("../../DynPRE-raw/DynPRE/DynPRE.py") 
    
    if not os.path.exists(dynpre_path):
        logging.warning(f"DynPRE not found at {dynpre_path}, falling back to heuristic simulation.")
        return simulate_dynpre_segmentation(messages) # 回退到旧方法
        
    runner = RealDynPRERunner(dynpre_path=dynpre_path)
    return runner.run(messages)

def simulate_neupre_segmentation(messages: List[bytes],
                                responses: List[bytes] = None,
                                ground_truth: List[List[int]] = None,
                                use_supervised: bool = True) -> List[List[int]]:
    """
    Use NeuPRE's approach for segmentation.

    Args:
        messages: Protocol messages
        responses: Response messages (for unsupervised learning)
        ground_truth: Ground truth boundaries (for supervised learning)
        use_supervised: Whether to use supervised learning

    Returns:
        List of boundary lists
    """
    # [修改3] 启用并增强无监督学习 (Information Bottleneck)
    logging.info("Initializing Unsupervised IB Learner...")
    from modules.format_learner import InformationBottleneckFormatLearner

    # 参数调优建议：
    # d_model: 增加到 128 或 256 以捕捉更复杂的二进制模式
    # beta: 信息瓶颈的关键参数。
    #       beta 越小 (如 1e-3)，压缩越少，保留更多细节（容易产生碎片化字段）
    #       beta 越大 (如 1e-1)，压缩越强，倾向于合并字段（容易丢失短字段）
    #       对于 Modbus 这种紧凑协议，建议 0.005 - 0.01
   
    # 使用原始无监督IB方法（改进参数）
    learner = InformationBottleneckFormatLearner(
        d_model=256,      # 增加模型容量
        nhead=8,
        num_layers=4,
        beta=0.002       # 设为 0.005 以平衡细节和整体结构
    )

    # 训练模型
    # 注意：真实逆向中没有 label，所以不仅不需要 train_gt，连 responses 也是可选的
    # 如果有 responses (对应请求的响应)，传入会有帮助；没有就传 None
    logging.info(f"Training on {len(messages)} messages (Unsupervised)...")
    learner.train(messages, responses if responses else messages, epochs=50, batch_size=32)

    # 1. 初始分割
    raw_segmentations = []
    # 2. [新增] 全局精细化 (Refinement)
    logging.info("Applying NeuPRE Global Alignment Refinement...")
    refiner = GlobalAlignmentRefiner(alignment_threshold=0.5) # 50% 投票权
    refined_segmentations = refiner.refine(messages, raw_segmentations)

    return refined_segmentations


def run_experiment2(num_samples: int = 100,
                   output_dir: str = './experiment2_results',
                   use_real_data: bool = True,
                   use_dynpre_ground_truth: bool = False):
    """
    Run Experiment 2: Field Boundary Accuracy.

    Args:
        num_samples: Number of samples per protocol
        output_dir: Output directory
        use_real_data: If True, use real pcap data; if False, use synthetic data
        use_dynpre_ground_truth: If True, use DynPRE's fine-grained ground truth for fair comparison
    """
    setup_logging(level=logging.INFO)
    logging.info("=" * 80)
    logging.info("EXPERIMENT 2: Field Boundary Accuracy")
    logging.info("=" * 80)

    evaluator = NeuPREEvaluator(output_dir=output_dir)

    # Test on different protocol types
    if use_real_data:
        protocols = {}

        if use_dynpre_ground_truth:
            # Use DynPRE's fine-grained ground truth for fair comparison
            logging.info("Using DynPRE Ground Truth for fair comparison")
            dynpre_loader = DynPREGroundTruthLoader(dynpre_output_dir='../../DynPRE/examples')

            # Load protocols that have DynPRE ground truth
            for protocol_name in dynpre_loader.get_available_protocols():
                try:
                    messages, ground_truth = dynpre_loader.load_ground_truth(protocol_name)
                    # Limit to num_samples
                    if len(messages) > num_samples:
                        messages = messages[:num_samples]
                        ground_truth = ground_truth[:num_samples]
                    protocols[protocol_name] = (messages, ground_truth)
                    logging.info(f"Loaded {len(messages)} {protocol_name.upper()} messages with DynPRE ground truth")
                except Exception as e:
                    logging.warning(f"Failed to load DynPRE ground truth for {protocol_name}: {e}")

            # For protocols without DynPRE ground truth, fall back to PCAP loader
            pcap_loader = PCAPDataLoader(data_dir='./data')
            for protocol_name in ['dhcp', 'dns']:
                if protocol_name not in protocols:
                    try:
                        messages, ground_truth = pcap_loader.load_protocol_data(protocol_name, max_messages=num_samples)
                        if messages:
                            protocols[protocol_name] = (messages, ground_truth)
                            logging.info(f"Loaded {len(messages)} {protocol_name.upper()} messages (protocol-spec ground truth)")
                    except Exception as e:
                        logging.warning(f"Failed to load {protocol_name}: {e}")
        else:
            # Use protocol specification ground truth from PCAP files
            logging.info("Using REAL protocol data from pcap files (protocol-spec ground truth)")
            pcap_loader = PCAPDataLoader(data_dir='./data')

            for protocol_name in ['dhcp', 'dns', 'modbus', 'smb2']:
                try:
                    messages, ground_truth = pcap_loader.load_protocol_data(protocol_name, max_messages=num_samples)
                    if messages:
                        protocols[protocol_name] = (messages, ground_truth)
                        logging.info(f"Loaded {len(messages)} {protocol_name.upper()} messages")
                except Exception as e:
                    logging.warning(f"Failed to load {protocol_name}: {e}")

    else:
        logging.info("Using SYNTHETIC protocol data")
        protocols = {
            'simple': ProtocolDataset.generate_simple_protocol(num_samples),
            'high_entropy': ProtocolDataset.generate_high_entropy_protocol(num_samples),
            'mixed': ProtocolDataset.generate_mixed_protocol(num_samples)
        }

    all_comparisons = []

    for protocol_name, (messages, ground_truth) in protocols.items():
        logging.info(f"\n{'='*80}")
        logging.info(f"Testing on {protocol_name} protocol")
        logging.info(f"{'='*80}")

        # NeuPRE segmentation
        # logging.info("Running NeuPRE segmentation (Unsupervised)...")
        
        # # [修改5] 关键修改：use_supervised=False
        # # 注意：这里我们依然传入 ground_truth，但仅用于该函数内部可能的"评估"（如果代码有的话），
        # # 或者干脆传 None，最安全。我们的 simulate 函数已经忽略了它。
        # neupre_boundaries = simulate_neupre_segmentation(
        #     messages,
        #     responses=None,
        #     ground_truth=None,    # 传 None 确保绝对不偷看答案
        #     use_supervised=False  # 强制关闭
        # )

        # # DYNpre segmentation
        # logging.info("Running DYNpre segmentation (Heuristic)...")
        # dynpre_boundaries = simulate_dynpre_segmentation(messages)
        # NeuPRE (带 Refinement)
        logging.info("Running NeuPRE segmentation (Unsupervised + Refinement)...")
        neupre_boundaries = simulate_neupre_segmentation(...)
        
        # DYNpre (调用真实脚本)
        logging.info("Running Real DYNpre segmentation...")
        try:
            dynpre_boundaries = run_real_dynpre(messages)
            
            # 检查 DYNpre 是否成功返回结果
            if not dynpre_boundaries:
                logging.warning("Real DYNpre failed, using empty boundaries.")
                dynpre_boundaries = [[0, len(m)] for m in messages]
            elif len(dynpre_boundaries) != len(messages):
                logging.warning("DYNpre count mismatch, truncating or padding.")
                dynpre_boundaries = dynpre_boundaries[:len(messages)]
                
        except Exception as e:
            logging.error(f"Critical error running DynPRE: {e}")
            dynpre_boundaries = [[0, len(m)] for m in messages]

        # Evaluate
        neupre_metrics = evaluator.evaluate_segmentation_accuracy(
            neupre_boundaries, ground_truth
        )

        dynpre_metrics = evaluator.evaluate_segmentation_accuracy(
            dynpre_boundaries, ground_truth
        )

        # Compare
        comparison = evaluator.compare_methods(
            neupre_metrics, dynpre_metrics, 'segmentation'
        )
        comparison['protocol'] = protocol_name
        all_comparisons.append(comparison)

        # Plot
        evaluator.plot_segmentation_comparison(
            neupre_metrics, dynpre_metrics,
            filename=f'segmentation_{protocol_name}.png'
        )

        # Log results
        logging.info(f"\n{protocol_name.upper()} Results:")
        logging.info(f"NeuPRE F1: {neupre_metrics.f1_score:.4f}")
        logging.info(f"DYNpre F1: {dynpre_metrics.f1_score:.4f}")
        logging.info(f"Improvement: {comparison['improvement'].get('f1_score', 0):.2f}%")

    # Overall summary
    evaluator.generate_report(all_comparisons, filename='experiment2_report.txt')
    evaluator.save_metrics_json(all_comparisons, filename='experiment2_metrics.json')

    # Aggregate statistics
    logging.info("\n" + "=" * 80)
    logging.info("EXPERIMENT 2 SUMMARY (AGGREGATED)")
    logging.info("=" * 80)

    avg_neupre_f1 = np.mean([c['neupre'].f1_score for c in all_comparisons])
    avg_dynpre_f1 = np.mean([c['dynpre'].f1_score for c in all_comparisons])
    avg_improvement = (avg_neupre_f1 - avg_dynpre_f1) / avg_dynpre_f1 * 100

    logging.info(f"Average NeuPRE F1-Score: {avg_neupre_f1:.4f}")
    logging.info(f"Average DYNpre F1-Score: {avg_dynpre_f1:.4f}")
    logging.info(f"Average Improvement: {avg_improvement:.2f}%")

    avg_neupre_perfect = np.mean([c['neupre'].perfect_match_rate for c in all_comparisons])
    avg_dynpre_perfect = np.mean([c['dynpre'].perfect_match_rate for c in all_comparisons])

    logging.info(f"Average NeuPRE Perfect Match: {avg_neupre_perfect:.4f}")
    logging.info(f"Average DYNpre Perfect Match: {avg_dynpre_perfect:.4f}")
    logging.info("=" * 80)


if __name__ == '__main__':
    # run_experiment2(
    #     num_samples=100,
    #     output_dir='./experiments/experiment2_results',
    #     use_real_data=True  # Set to True to use real pcap data, False for synthetic
    # )
    run_experiment2(
        num_samples=1000,          # 真实学习需要更多样本，建议增加到 1000
        output_dir='./experiments/experiment2_results',
        use_real_data=True,        # 必须为 True
        use_dynpre_ground_truth=False # 使用 PCAP 自带的标准作为验证基准
    )
