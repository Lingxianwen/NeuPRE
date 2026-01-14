#!/usr/bin/env python3
"""
改进版NetPlier - 真实数据集版本
主要改进：
1. 使用真实数据集（../Msg2/）
2. 智能字段合并策略
3. 改进的概率推理
4. 协议特异性优化
5. 后处理边界调整
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import random
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from pathlib import Path
import argparse
from sklearn.metrics import f1_score, accuracy_score
import warnings
import json
import ast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class Message:
    """消息类"""

    def __init__(self, data, source=None, destination=None, timestamp=None):
        self.data = data if isinstance(data, bytes) else bytes.fromhex(data.replace(' ', ''))
        self.source = source or "0.0.0.0:0"
        self.destination = destination or "0.0.0.0:0"
        self.timestamp = timestamp or 0
        self.id = random.randint(1000000, 9999999)


class ImprovedNetPlierDataLoader:
    """改进的数据加载器 - 真实数据集版本"""

    def __init__(self, data_root: str = "../../Msg2"):
        self.data_root = Path(data_root)
        self.csv_root = self.data_root / "csv"
        self.supported_protocols = [
            'smb', 'smb2', 'dns', 's7comm', 'dnp3',
            'modbus', 'ftp', 'tls', 'dhcp'
        ]

    def load_protocol_data(self, protocol_name: str) -> List[Dict]:
        """从真实CSV文件加载协议数据"""
        logger.info(f"📊 加载 {protocol_name.upper()} 协议数据...")

        # 构建CSV文件路径
        csv_path = self.csv_root / protocol_name / f"{protocol_name}.csv"

        if not csv_path.exists():
            logger.warning(f"   ❌ CSV文件不存在: {csv_path}")
            return []

        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            logger.info(f"   📁 成功读取CSV文件: {csv_path}")
            logger.info(f"   📊 数据行数: {len(df)}")

            # 转换为标准格式
            data = self._convert_csv_to_standard_format(df, protocol_name)
            logger.info(f"   ✅ 成功转换 {len(data)} 条数据")

            return data

        except Exception as e:
            logger.error(f"   ❌ 加载CSV文件失败: {e}")
            return []

    def _convert_csv_to_standard_format(self, df: pd.DataFrame, protocol_name: str) -> List[Dict]:
        """将CSV数据转换为标准格式"""
        data = []

        for index, row in df.iterrows():
            try:
                # 获取基本信息
                hex_data = str(row['HexData']).strip()
                length = int(row['Length'])

                # 解析边界信息
                boundaries_str = str(row['Boundaries']).strip()
                if boundaries_str and boundaries_str != 'nan':
                    # 解析边界字符串，如 "2,3,5,7,9,10,11,12,13,14,15,16"
                    boundaries = [int(x.strip()) for x in boundaries_str.split(',')]
                    # 确保边界包含起始位置和结束位置
                    if 0 not in boundaries:
                        boundaries = [0] + boundaries
                    if length not in boundaries:
                        boundaries.append(length)
                    boundaries = sorted(list(set(boundaries)))
                else:
                    # 如果没有边界信息，使用默认边界
                    boundaries = [0, length]

                # 验证hex数据
                try:
                    raw_bytes = bytes.fromhex(hex_data.replace(' ', ''))
                    actual_length = len(raw_bytes)

                    # 调整边界以匹配实际长度
                    adjusted_boundaries = [b for b in boundaries if b <= actual_length]
                    if actual_length not in adjusted_boundaries:
                        adjusted_boundaries.append(actual_length)

                    sample = {
                        'raw_data': hex_data,
                        'protocol': protocol_name,
                        'bytes': raw_bytes,
                        'length': actual_length,
                        'message_type': row.get('FunctionCode', 'unknown'),
                        'ground_truth_boundaries': sorted(adjusted_boundaries),
                        'original_index': index
                    }
                    data.append(sample)

                except ValueError as e:
                    logger.warning(f"   ⚠️ 跳过无效hex数据 (行{index}): {e}")
                    continue

            except Exception as e:
                logger.warning(f"   ⚠️ 跳过异常数据 (行{index}): {e}")
                continue

        return data

    def list_available_protocols(self) -> List[str]:
        """列出可用的协议"""
        available_protocols = []

        if self.csv_root.exists():
            for protocol_dir in self.csv_root.iterdir():
                if protocol_dir.is_dir():
                    csv_file = protocol_dir / f"{protocol_dir.name}.csv"
                    if csv_file.exists():
                        available_protocols.append(protocol_dir.name)

        return available_protocols


class ImprovedNetPlierAlgorithm:
    """改进的NetPlier算法"""

    def __init__(self):
        self.min_field_size = 1
        self.max_field_size = 64
        self.merge_threshold = 2  # 合并小字段的阈值
        self.boundary_tolerance = 1  # 边界容错

        # 协议特异性参数
        self.protocol_params = {
            'dns': {'min_field_size': 2, 'merge_threshold': 1},
            'modbus': {'min_field_size': 1, 'merge_threshold': 2},
            'smb': {'min_field_size': 1, 'merge_threshold': 4},
            'smb2': {'min_field_size': 1, 'merge_threshold': 4},
            'dhcp': {'min_field_size': 1, 'merge_threshold': 3},
            'dnp3': {'min_field_size': 1, 'merge_threshold': 2}
        }

    def extract_fields(self, messages: List[Message], protocol_name: str = None) -> List[List[int]]:
        """提取字段边界 - 改进版"""
        logger.info(f"🔍 改进版NetPlier分析 {len(messages)} 个消息...")

        # 应用协议特异性参数
        if protocol_name and protocol_name in self.protocol_params:
            params = self.protocol_params[protocol_name]
            self.min_field_size = params.get('min_field_size', self.min_field_size)
            self.merge_threshold = params.get('merge_threshold', self.merge_threshold)

        # 步骤1: 智能多序列比对
        aligned_messages = self._intelligent_msa(messages)

        # 步骤2: 生成初始字段候选
        initial_candidates = self._generate_smart_candidates(aligned_messages)

        # 步骤3: 改进的概率推理
        keyword_field = self._improved_probabilistic_inference(initial_candidates, aligned_messages)

        # 步骤4: 基于关键字段聚类
        clusters = self._cluster_by_keyword(aligned_messages, keyword_field, initial_candidates)

        # 步骤5: 生成并优化边界
        raw_boundaries = self._generate_initial_boundaries(clusters, aligned_messages)

        # 步骤6: 智能后处理
        final_boundaries = self._intelligent_postprocessing(raw_boundaries, aligned_messages, protocol_name)

        return final_boundaries

    def _intelligent_msa(self, messages: List[Message]) -> List[bytes]:
        """智能多序列比对"""
        logger.info("   执行智能多序列比对...")

        # 按长度分组处理
        length_groups = defaultdict(list)
        for msg in messages:
            length_groups[len(msg.data)].append(msg.data)

        # 找到主要长度组
        main_length = max(length_groups.keys(), key=lambda x: len(length_groups[x]))
        main_group = length_groups[main_length]

        # 对主要组进行对齐
        aligned = []
        for msg in messages:
            if len(msg.data) == main_length:
                aligned.append(msg.data)
            elif len(msg.data) < main_length:
                # 智能填充
                padded = msg.data + b'\x00' * (main_length - len(msg.data))
                aligned.append(padded)
            else:
                # 智能截断（保留前缀）
                aligned.append(msg.data[:main_length])

        return aligned

    def _generate_smart_candidates(self, aligned_messages: List[bytes]) -> List[Tuple[int, int]]:
        """生成智能字段候选"""
        logger.info("   生成智能字段候选...")

        if not aligned_messages:
            return []

        length = len(aligned_messages[0])
        candidates = []

        # 分析字节熵和变化模式
        entropy_scores = []
        change_points = []

        for pos in range(length):
            values = [msg[pos] for msg in aligned_messages]
            # 计算熵
            value_counts = Counter(values)
            entropy = -sum((count / len(values)) * np.log2(count / len(values))
                           for count in value_counts.values() if count > 0)
            entropy_scores.append(entropy)

            # 检测变化点
            if pos > 0:
                prev_values = [msg[pos - 1] for msg in aligned_messages]
                change_ratio = sum(1 for v1, v2 in zip(prev_values, values) if v1 != v2) / len(values)
                if change_ratio > 0.3:  # 30%以上的消息在此位置有变化
                    change_points.append(pos)

        # 基于熵阈值检测边界
        entropy_threshold = np.mean(entropy_scores) + np.std(entropy_scores) * 0.5
        entropy_boundaries = [i for i, entropy in enumerate(entropy_scores) if entropy > entropy_threshold]

        # 合并边界候选
        all_boundaries = sorted(set([0] + change_points + entropy_boundaries + [length]))

        # 生成字段候选
        for i in range(len(all_boundaries) - 1):
            start = all_boundaries[i]
            end = all_boundaries[i + 1]
            if end - start >= self.min_field_size:
                candidates.append((start, end))

        # 添加常见字段长度的候选
        common_sizes = [1, 2, 4, 8, 16]
        for size in common_sizes:
            for start in range(0, length - size, max(1, size // 2)):
                end = start + size
                if end <= length:
                    candidates.append((start, end))

        # 去重并排序
        candidates = sorted(list(set(candidates)))
        logger.info(f"   生成了 {len(candidates)} 个智能候选")
        return candidates

    def _improved_probabilistic_inference(self, candidates: List[Tuple[int, int]],
                                          aligned_messages: List[bytes]) -> int:
        """改进的概率推理"""
        logger.info("   执行改进的概率推理...")

        if not candidates:
            return 0

        best_score = -1
        best_field = 0

        for i, (start, end) in enumerate(candidates):
            field_values = []
            for msg in aligned_messages:
                if end <= len(msg):
                    field_values.append(msg[start:end])

            if not field_values:
                continue

            # 改进的评分函数
            score = self._calculate_improved_field_score(field_values, start, end, len(aligned_messages))

            if score > best_score:
                best_score = score
                best_field = i

        logger.info(f"   选择字段 {best_field} 作为关键字段，得分: {best_score:.3f}")
        return best_field

    def _calculate_improved_field_score(self, field_values: List[bytes], start: int, end: int,
                                        total_msgs: int) -> float:
        """计算改进的字段得分"""
        length = end - start
        unique_values = len(set(field_values))
        total_values = len(field_values)

        # 1. 多样性得分（改进）
        if unique_values == 1:
            diversity_score = 0.1  # 常量字段不适合做关键字
        elif unique_values == total_values:
            diversity_score = 0.3  # 完全随机也不理想
        else:
            # 理想的多样性在2-8个不同值之间
            ideal_diversity = min(8, max(2, total_values // 10))
            diversity_score = 1.0 - abs(unique_values - ideal_diversity) / ideal_diversity
            diversity_score = max(0.1, diversity_score)

        # 2. 位置得分（关键字段通常在前面）
        position_score = 1.0 / (start + 1) if start < 10 else 0.1

        # 3. 长度得分（改进）
        if length == 1:
            length_score = 0.9  # 单字节字段很适合做关键字
        elif length == 2:
            length_score = 1.0  # 双字节字段最理想
        elif length <= 4:
            length_score = 0.7
        elif length <= 8:
            length_score = 0.4
        else:
            length_score = 0.1

        # 4. 分布均匀性得分
        value_counts = Counter(field_values)
        max_count = max(value_counts.values())
        min_count = min(value_counts.values())
        if max_count > 0:
            distribution_score = min_count / max_count
        else:
            distribution_score = 0.1

        # 5. 语义得分（基于常见模式）
        semantic_score = self._calculate_semantic_score(field_values, start)

        # 综合得分
        total_score = (diversity_score * 0.3 +
                       position_score * 0.2 +
                       length_score * 0.25 +
                       distribution_score * 0.15 +
                       semantic_score * 0.1)

        return total_score

    def _calculate_semantic_score(self, field_values: List[bytes], start: int) -> float:
        """计算语义得分"""
        # 检查是否符合常见的关键字段模式
        score = 0.5  # 基础分数

        # 如果在开头，可能是协议标识
        if start == 0:
            score += 0.2

        # 检查是否有明显的枚举值模式
        unique_values = set(field_values)
        if len(unique_values) <= 10 and len(field_values) >= 20:
            score += 0.2

        # 检查数值范围
        try:
            int_values = [int.from_bytes(v, 'big') for v in field_values if v]
            if int_values:
                value_range = max(int_values) - min(int_values)
                if value_range < 256:  # 较小的数值范围
                    score += 0.1
        except:
            pass

        return min(1.0, score)

    def _cluster_by_keyword(self, aligned_messages: List[bytes], keyword_field: int,
                            candidates: List[Tuple[int, int]]) -> Dict[bytes, List[int]]:
        """基于关键字段聚类"""
        logger.info("   基于关键字段聚类...")

        clusters = defaultdict(list)

        if keyword_field < len(candidates):
            start, end = candidates[keyword_field]
            for i, msg in enumerate(aligned_messages):
                if end <= len(msg):
                    key = msg[start:end]
                    clusters[key].append(i)
        else:
            # 回退到简单聚类
            for i, msg in enumerate(aligned_messages):
                key = msg[:min(2, len(msg))]
                clusters[key].append(i)

        logger.info(f"   生成了 {len(clusters)} 个聚类")
        return clusters

    def _generate_initial_boundaries(self, clusters: Dict[bytes, List[int]],
                                     aligned_messages: List[bytes]) -> List[List[int]]:
        """生成初始边界"""
        logger.info("   生成初始边界...")

        boundaries_list = []

        for i, msg in enumerate(aligned_messages):
            boundaries = self._detect_boundaries_for_message(msg, i, clusters, aligned_messages)
            boundaries_list.append(boundaries)

        return boundaries_list

    def _detect_boundaries_for_message(self, msg: bytes, msg_idx: int,
                                       clusters: Dict[bytes, List[int]],
                                       all_messages: List[bytes]) -> List[int]:
        """为单个消息检测边界"""
        boundaries = [0]

        # 基于字节变化检测边界
        for pos in range(1, len(msg)):
            # 检查当前位置是否应该是边界
            boundary_score = 0

            # 1. 字节值变化
            if pos < len(msg) - 1:
                curr_byte = msg[pos]
                prev_byte = msg[pos - 1]
                next_byte = msg[pos + 1]

                if abs(curr_byte - prev_byte) > 30:
                    boundary_score += 0.3
                if abs(next_byte - curr_byte) > 30:
                    boundary_score += 0.3

            # 2. 模式变化
            if pos >= 2 and pos < len(msg) - 2:
                left_pattern = msg[pos - 2:pos]
                right_pattern = msg[pos:pos + 2]
                if left_pattern != right_pattern:
                    boundary_score += 0.2

            # 3. 对齐位置（2,4,8字节对齐）
            if pos % 2 == 0:
                boundary_score += 0.1
            if pos % 4 == 0:
                boundary_score += 0.1

            # 4. 与其他消息的一致性
            consistency_score = self._check_boundary_consistency(pos, msg_idx, all_messages)
            boundary_score += consistency_score * 0.3

            if boundary_score > 0.5:
                boundaries.append(pos)

        return sorted(boundaries)

    def _check_boundary_consistency(self, pos: int, msg_idx: int, all_messages: List[bytes]) -> float:
        """检查边界一致性"""
        if pos >= len(all_messages[msg_idx]):
            return 0

        consistent_count = 0
        total_count = 0

        current_byte = all_messages[msg_idx][pos]

        for i, other_msg in enumerate(all_messages):
            if i != msg_idx and pos < len(other_msg):
                total_count += 1
                if other_msg[pos] == current_byte:
                    consistent_count += 1

        return consistent_count / total_count if total_count > 0 else 0

    def _intelligent_postprocessing(self, raw_boundaries: List[List[int]],
                                    aligned_messages: List[bytes],
                                    protocol_name: str = None) -> List[List[int]]:
        """智能后处理"""
        logger.info("   执行智能后处理...")

        processed_boundaries = []

        for i, boundaries in enumerate(raw_boundaries):
            # 步骤1: 合并过小的字段
            merged = self._merge_small_fields(boundaries)

            # 步骤2: 调整边界到对齐位置
            aligned = self._align_boundaries(merged, aligned_messages[i])

            # 步骤3: 应用协议特异性规则
            if protocol_name:
                aligned = self._apply_protocol_rules(aligned, protocol_name, aligned_messages[i])

            # 步骤4: 最终验证和清理
            final = self._validate_and_clean_boundaries(aligned, aligned_messages[i])

            processed_boundaries.append(final)

        logger.info("   后处理完成")
        return processed_boundaries

    def _merge_small_fields(self, boundaries: List[int]) -> List[int]:
        """合并过小的字段"""
        if len(boundaries) <= 2:
            return boundaries

        merged = [boundaries[0]]

        for i in range(1, len(boundaries)):
            # 计算字段大小
            field_size = boundaries[i] - merged[-1]

            if field_size < self.merge_threshold:
                # 跳过这个边界，实现合并
                continue
            else:
                merged.append(boundaries[i])

        return merged

    def _align_boundaries(self, boundaries: List[int], message: bytes) -> List[int]:
        """将边界对齐到合适的位置"""
        aligned = [boundaries[0]]  # 保持起始位置

        for boundary in boundaries[1:]:
            # 尝试对齐到2字节边界
            if boundary % 2 == 1 and boundary + 1 < len(message):
                aligned_boundary = boundary + 1
            else:
                aligned_boundary = boundary

            # 确保不与前一个边界过于接近
            if aligned_boundary - aligned[-1] >= self.min_field_size:
                aligned.append(aligned_boundary)

        return aligned

    def _apply_protocol_rules(self, boundaries: List[int], protocol_name: str, message: bytes) -> List[int]:
        """应用协议特异性规则"""

        if protocol_name == 'dns':
            # DNS协议：确保头部12字节完整
            if 12 not in boundaries and 12 < len(message):
                boundaries = sorted(boundaries + [12])

        elif protocol_name == 'modbus':
            # Modbus协议：确保MBAP头部7字节
            if 7 not in boundaries and 7 < len(message):
                boundaries = sorted(boundaries + [7])

        elif protocol_name in ['smb', 'smb2']:
            # SMB协议：确保头部字段
            important_positions = [4, 8]
            for pos in important_positions:
                if pos not in boundaries and pos < len(message):
                    boundaries = sorted(boundaries + [pos])

        elif protocol_name == 'dnp3':
            # DNP3协议：确保头部字段
            important_positions = [2, 10]
            for pos in important_positions:
                if pos not in boundaries and pos < len(message):
                    boundaries = sorted(boundaries + [pos])

        return boundaries

    def _validate_and_clean_boundaries(self, boundaries: List[int], message: bytes) -> List[int]:
        """验证和清理边界"""
        # 确保边界在有效范围内
        valid_boundaries = [b for b in boundaries if 0 <= b < len(message)]

        # 确保包含起始位置
        if 0 not in valid_boundaries:
            valid_boundaries.insert(0, 0)

        # 移除重复边界
        valid_boundaries = sorted(list(set(valid_boundaries)))

        # 限制字段数量（避免过度分割）
        max_fields = min(10, len(message) // 2)
        if len(valid_boundaries) > max_fields:
            # 保留最重要的边界
            valid_boundaries = valid_boundaries[:max_fields]

        return valid_boundaries


class ImprovedNetPlierEvaluator:
    """改进的评估器"""

    def __init__(self):
        pass

    def evaluate_boundaries(self, predicted_boundaries: List[int],
                            ground_truth_boundaries: List[int],
                            sequence_length: int) -> Dict[str, float]:
        """评估边界检测性能"""
        return self._standard_evaluation(predicted_boundaries, ground_truth_boundaries, sequence_length)

    def _standard_evaluation(self, predicted_boundaries: List[int],
                             ground_truth_boundaries: List[int],
                             sequence_length: int) -> Dict[str, float]:
        """标准评估"""
        # 创建位置标记
        pred_positions = set(predicted_boundaries)
        true_positions = set(ground_truth_boundaries)

        # 计算准确率
        correct_positions = 0
        for pos in range(sequence_length):
            pred_is_boundary = pos in pred_positions
            true_is_boundary = pos in true_positions
            if pred_is_boundary == true_is_boundary:
                correct_positions += 1

        accuracy = correct_positions / sequence_length if sequence_length > 0 else 0

        # 计算精确率和召回率
        if len(predicted_boundaries) > 0:
            true_positives = len(true_positions & pred_positions)
            precision = true_positives / len(predicted_boundaries)
        else:
            precision = 0

        if len(ground_truth_boundaries) > 0:
            true_positives = len(true_positions & pred_positions)
            recall = true_positives / len(ground_truth_boundaries)
        else:
            recall = 0

        # F1分数
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0

        # 完美匹配率
        pred_fields = self._boundaries_to_fields(predicted_boundaries, sequence_length)
        true_fields = self._boundaries_to_fields(ground_truth_boundaries, sequence_length)

        if len(true_fields) > 0:
            perfect_matches = len(set(pred_fields) & set(true_fields))
            perfection = perfect_matches / len(true_fields)
        else:
            perfection = 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'perfection': perfection
        }

    def _boundaries_to_fields(self, boundaries: List[int], length: int) -> List[Tuple[int, int]]:
        """将边界转换为字段"""
        if not boundaries:
            return [(0, length)] if length > 0 else []

        fields = []
        boundaries = sorted(set(boundaries))

        for i in range(len(boundaries)):
            start = boundaries[i]
            if i < len(boundaries) - 1:
                end = boundaries[i + 1]
            else:
                end = length

            if start < end and start < length:
                fields.append((start, min(end, length)))

        return fields


class ImprovedNetPlierExperiment:
    """改进的实验管理器"""

    def __init__(self, data_root: str = "../../Msg2"):
        self.data_loader = ImprovedNetPlierDataLoader(data_root)
        self.algorithm = ImprovedNetPlierAlgorithm()
        self.evaluator = ImprovedNetPlierEvaluator()

        # 获取可用协议
        self.available_protocols = self.data_loader.list_available_protocols()
        if not self.available_protocols:
            logger.warning("❌ 未找到可用的协议数据")
            self.available_protocols = []

        self.results = {}

    def run_experiments(self, protocols: List[str] = None, max_samples: int = None):
        """运行实验"""

        if protocols is None:
            protocols = self.available_protocols

        # 过滤出可用的协议
        protocols = [p for p in protocols if p in self.available_protocols]

        if not protocols:
            logger.error("❌ 没有可用的协议进行测试")
            return

        logger.info("🚀 改进版NetPlier实验开始")
        logger.info("=" * 70)

        for protocol in protocols:
            logger.info(f"\n📊 测试协议: {protocol.upper()}")
            logger.info("-" * 50)

            # 加载数据
            data = self.data_loader.load_protocol_data(protocol)

            if not data:
                logger.warning(f"   ❌ 跳过 {protocol}: 无数据")
                continue

            # 限制样本数量
            if max_samples and len(data) > max_samples:
                data = random.sample(data, max_samples)
                logger.info(f"   📝 限制样本数量: {max_samples}")

            # 转换为Message对象
            messages = []
            for sample in data:
                msg = Message(sample['raw_data'])
                messages.append(msg)

            try:
                # 运行改进的NetPlier算法
                logger.info(f"   🔍 运行改进版NetPlier算法...")
                predicted_boundaries = self.algorithm.extract_fields(messages, protocol)

                # 评估性能
                logger.info(f"   📈 评估性能...")
                all_metrics = []

                for sample, pred_boundaries in zip(data, predicted_boundaries):
                    true_boundaries = sample['ground_truth_boundaries']
                    length = sample['length']

                    metrics = self.evaluator.evaluate_boundaries(pred_boundaries, true_boundaries, length)
                    all_metrics.append(metrics)

                # 计算平均指标
                avg_metrics = {}
                for key in ['accuracy', 'precision', 'recall', 'f1_score', 'perfection']:
                    values = [m[key] for m in all_metrics if not np.isnan(m[key])]
                    avg_metrics[key] = np.mean(values) if values else 0.0

                # 保存结果
                self.results[protocol] = {
                    'csv_rows': len(data),
                    'metrics': avg_metrics
                }

                # 显示结果
                logger.info(f"   ✅ 结果:")
                logger.info(f"      CSV行数: {len(data)}")
                logger.info(f"      准确率: {avg_metrics['accuracy']:.4f}")
                logger.info(f"      精确率: {avg_metrics['precision']:.4f}")
                logger.info(f"      召回率: {avg_metrics['recall']:.4f}")
                logger.info(f"      F1分数: {avg_metrics['f1_score']:.4f}")
                logger.info(f"      完美率: {avg_metrics['perfection']:.4f}")

            except Exception as e:
                logger.error(f"   ❌ 处理 {protocol} 时出错: {e}")
                self.results[protocol] = {
                    'csv_rows': len(data),
                    'metrics': {'accuracy': 0, 'precision': 0, 'recall': 0,
                                'f1_score': 0, 'perfection': 0},
                    'error': str(e)
                }

    def generate_report(self):
        """生成报告"""
        logger.info(f"\n" + "=" * 70)
        logger.info("📊 改进版NetPlier实验报告")
        logger.info("=" * 70)

        if not self.results:
            logger.warning("❌ 没有实验结果")
            return

        # 创建结果表格
        report_data = []
        for protocol, result in self.results.items():
            metrics = result['metrics']
            report_data.append({
                'Protocol': protocol.upper(),
                'CSV_Rows': result['csv_rows'],
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-score': f"{metrics['f1_score']:.4f}",
                'Perfection': f"{metrics['perfection']:.4f}"
            })

        # 显示表格
        df = pd.DataFrame(report_data)
        print("\n实验结果表格:")
        print(df.to_string(index=False))

        # 计算总体统计
        logger.info(f"\n🎯 总体统计:")
        total_samples = sum(r['csv_rows'] for r in self.results.values())
        avg_accuracy = np.mean([r['metrics']['accuracy'] for r in self.results.values()])
        avg_precision = np.mean([r['metrics']['precision'] for r in self.results.values()])
        avg_recall = np.mean([r['metrics']['recall'] for r in self.results.values()])
        avg_f1 = np.mean([r['metrics']['f1_score'] for r in self.results.values()])
        avg_perfection = np.mean([r['metrics']['perfection'] for r in self.results.values()])

        logger.info(f"   总样本数: {total_samples}")
        logger.info(f"   平均准确率: {avg_accuracy:.4f}")
        logger.info(f"   平均精确率: {avg_precision:.4f}")
        logger.info(f"   平均召回率: {avg_recall:.4f}")
        logger.info(f"   平均F1分数: {avg_f1:.4f}")
        logger.info(f"   平均完美率: {avg_perfection:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='改进版NetPlier实验 - 真实数据集版本')

    parser.add_argument('--data-root', type=str, default="../../Msg2",
                        help='数据集根目录')

    parser.add_argument('--protocols', nargs='+',
                        help='要测试的协议列表')

    parser.add_argument('--max-samples', type=int, default=None,
                        help='每个协议的最大样本数')

    parser.add_argument('--list-protocols', action='store_true',
                        help='列出可用的协议')

    args = parser.parse_args()

    # 创建实验管理器
    experiment = ImprovedNetPlierExperiment(args.data_root)

    # 列出可用协议
    if args.list_protocols:
        available_protocols = experiment.data_loader.list_available_protocols()
        logger.info(f"📋 可用协议: {available_protocols}")
        return

    logger.info(f"🌟 改进版NetPlier实验设置:")
    logger.info(f"   数据根目录: {args.data_root}")
    logger.info(f"   可用协议: {experiment.available_protocols}")
    logger.info(f"   测试协议: {args.protocols or 'ALL'}")
    logger.info(f"   最大样本: {args.max_samples or 'UNLIMITED'}")

    # 运行实验
    experiment.run_experiments(protocols=args.protocols, max_samples=args.max_samples)

    # 生成报告
    experiment.generate_report()

    logger.info("\n✅ 改进版NetPlier实验完成！")
    logger.info("\n🎉 主要特性:")
    logger.info("   1. 使用真实数据集")
    logger.info("   2. 智能字段合并策略")
    logger.info("   3. 改进的概率推理")
    logger.info("   4. 协议特异性优化")
    logger.info("   5. 智能后处理")


if __name__ == "__main__":
    main()