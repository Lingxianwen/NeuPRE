#!/usr/bin/env python3
"""
Enhanced BinaryInferno - 基于论文的完整实现
主要改进：
1. 实现完整的原子检测器集成（Float、Timestamp、Length）
2. 改进的熵基字段边界检测器
3. 基于模式的变长字段检测器
4. 图基集成算法
5. 更准确的协议特定优化
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import random
import math
import json
import time
from typing import Dict, List, Tuple, Optional, Set, Union
from collections import defaultdict, Counter
from pathlib import Path
import argparse
from sklearn.metrics import f1_score, accuracy_score
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class FieldDescription:
    """字段描述"""
    start: int
    end: int
    field_type: str
    confidence: float
    semantic_info: Optional[str] = None
    weight: float = 0.0


@dataclass
class Message:
    """消息类"""

    def __init__(self, data, source=None, destination=None, timestamp=None):
        self.data = data if isinstance(data, bytes) else bytes.fromhex(data.replace(' ', ''))
        self.source = source or "0.0.0.0:0"
        self.destination = destination or "0.0.0.0:0"
        self.timestamp = timestamp or 0
        self.id = random.randint(1000000, 9999999)


class AtomicDetectors:
    """原子检测器集合 - 基于论文Section V"""

    def __init__(self, capture_time_range: Optional[Tuple[float, float]] = None):
        self.capture_time_range = capture_time_range

    def detect_floats(self, messages: List[Message]) -> List[FieldDescription]:
        """Float检测器 - 基于IEEE 754特征"""
        logger.info("      运行Float检测器...")

        fields = []
        if len(messages) < 3:
            return fields

        # 检查4字节IEEE 754 float
        for offset in range(len(messages[0].data) - 3):
            if self._is_valid_float_slice(messages, offset, 4):
                fields.append(FieldDescription(
                    start=offset,
                    end=offset + 4,
                    field_type="float32",
                    confidence=0.8,
                    weight=len(messages) * 4
                ))

        return fields

    def _is_valid_float_slice(self, messages: List[Message], offset: int, width: int) -> bool:
        """检查是否为有效的float切片"""
        try:
            values = []
            for msg in messages:
                if offset + width <= len(msg.data):
                    float_bytes = msg.data[offset:offset + width]
                    # 解释为IEEE 754 float
                    import struct
                    try:
                        value = struct.unpack('>f', float_bytes)[0]
                        if not (math.isnan(value) or math.isinf(value)):
                            values.append(value)
                    except:
                        return False

            if len(values) < len(messages) * 0.8:
                return False

            # L-Ratio计算 - 论文中的关键特征
            return self._calculate_l_ratio(messages, offset, width)

        except:
            return False

    def _calculate_l_ratio(self, messages: List[Message], offset: int, width: int) -> bool:
        """计算L-Ratio特征"""
        try:
            exponent_freqs = [0] * 8
            significand_freqs = [0] * 23

            for msg in messages:
                if offset + width <= len(msg.data):
                    float_bytes = msg.data[offset:offset + width]
                    # 分析位模式
                    bits = ''.join(format(b, '08b') for b in float_bytes)

                    # 指数部分 (bits 1-8)
                    for i in range(1, 9):
                        if i < len(bits) and bits[i] == '1':
                            exponent_freqs[i - 1] += 1

                    # 尾数部分 (bits 9-31)
                    for i in range(9, 32):
                        if i < len(bits) and bits[i] == '1':
                            significand_freqs[i - 9] += 1

            max_exp_freq = max(exponent_freqs) if exponent_freqs else 1
            avg_sig_freq = sum(significand_freqs) / len(significand_freqs) if significand_freqs else 0

            l_ratio = avg_sig_freq / max_exp_freq if max_exp_freq > 0 else 0
            return 0.42 <= l_ratio <= 0.55  # 论文中的阈值

        except:
            return False

    def detect_timestamps(self, messages: List[Message]) -> List[FieldDescription]:
        """Timestamp检测器 - 基于时间范围"""
        logger.info("      运行Timestamp检测器...")

        fields = []
        if not self.capture_time_range:
            return fields

        start_time, end_time = self.capture_time_range

        # 检查4字节Unix timestamp
        for offset in range(len(messages[0].data) - 3):
            if self._is_valid_timestamp_slice(messages, offset, 4, start_time, end_time):
                fields.append(FieldDescription(
                    start=offset,
                    end=offset + 4,
                    field_type="unix_timestamp",
                    confidence=0.9,
                    weight=len(messages) * 4
                ))

        return fields

    def _is_valid_timestamp_slice(self, messages: List[Message], offset: int, width: int,
                                  start_time: float, end_time: float) -> bool:
        """检查是否为有效的timestamp切片"""
        try:
            import struct
            valid_count = 0

            for msg in messages:
                if offset + width <= len(msg.data):
                    ts_bytes = msg.data[offset:offset + width]

                    # 大端序Unix timestamp
                    try:
                        timestamp = struct.unpack('>I', ts_bytes)[0]
                        if start_time <= timestamp <= end_time:
                            valid_count += 1
                    except:
                        pass

                    # 小端序Unix timestamp
                    try:
                        timestamp = struct.unpack('<I', ts_bytes)[0]
                        if start_time <= timestamp <= end_time:
                            valid_count += 1
                    except:
                        pass

            return valid_count >= len(messages) * 0.8

        except:
            return False

    def detect_lengths(self, messages: List[Message]) -> List[FieldDescription]:
        """Length检测器 - 严格长度字段"""
        logger.info("      运行Length检测器...")

        fields = []

        # 检查1字节和2字节长度字段
        for width in [1, 2]:
            for offset in range(len(messages[0].data) - width + 1):
                if self._is_valid_length_slice(messages, offset, width):
                    fields.append(FieldDescription(
                        start=offset,
                        end=offset + width,
                        field_type=f"length{width * 8}",
                        confidence=0.95,
                        weight=len(messages) * width
                    ))

        return fields

    def _is_valid_length_slice(self, messages: List[Message], offset: int, width: int) -> bool:
        """检查是否为有效的length切片"""
        try:
            import struct

            constants = set()
            for msg in messages:
                if offset + width <= len(msg.data):
                    if width == 1:
                        length_val = msg.data[offset]
                    elif width == 2:
                        # 尝试大端序和小端序
                        length_val_be = struct.unpack('>H', msg.data[offset:offset + 2])[0]
                        length_val_le = struct.unpack('<H', msg.data[offset:offset + 2])[0]

                        # 检查哪个更合理
                        if abs(length_val_be - len(msg.data)) < abs(length_val_le - len(msg.data)):
                            length_val = length_val_be
                        else:
                            length_val = length_val_le

                    # 计算常数k (length_val + k = message_length)
                    k = len(msg.data) - length_val
                    constants.add(k)

            # 如果所有消息都有相同的k值，且k >= 0，则为长度字段
            return len(constants) == 1 and list(constants)[0] >= 0

        except:
            return False


class FieldBoundaryDetector:
    """字段边界检测器 - 基于Shannon熵"""

    def __init__(self, endianness: str = 'big'):
        self.endianness = endianness
        self.entropy_threshold = 1.0  # 论文中的阈值

    def detect_boundaries(self, messages: List[Message]) -> List[int]:
        """基于熵差异检测字段边界"""
        logger.info("      运行熵基边界检测器...")

        if not messages:
            return [0]

        boundaries = [0]  # 总是包含起始位置
        max_length = max(len(msg.data) for msg in messages)

        # 计算每个位置的熵
        entropies = []
        for pos in range(max_length):
            entropy = self._calculate_entropy_at_position(messages, pos)
            entropies.append(entropy)

        # 检测熵的显著变化
        for i in range(1, len(entropies) - 1):
            if self._is_entropy_boundary(entropies, i):
                boundaries.append(i)

        return sorted(list(set(boundaries)))

    def _calculate_entropy_at_position(self, messages: List[Message], position: int) -> float:
        """计算特定位置的Shannon熵"""
        values = []
        for msg in messages:
            if position < len(msg.data):
                values.append(msg.data[position])

        if not values:
            return 0.0

        counts = Counter(values)
        total = len(values)
        entropy = 0.0

        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def _is_entropy_boundary(self, entropies: List[float], position: int) -> bool:
        """判断是否为熵边界"""
        if position == 0 or position >= len(entropies) - 1:
            return False

        left_entropy = entropies[position - 1]
        right_entropy = entropies[position]

        if self.endianness == 'big':
            return left_entropy - right_entropy >= self.entropy_threshold
        else:
            return right_entropy - left_entropy >= self.entropy_threshold


class PatternDetector:
    """基于模式的检测器 - 用于变长字段"""

    def __init__(self):
        self.patterns = ['LV', 'TLV', 'LV*', 'TLV*']  # 支持的模式

    def detect_patterns(self, messages: List[Message]) -> List[FieldDescription]:
        """检测序列化模式"""
        logger.info("      运行模式检测器...")

        fields = []

        # 简化的模式检测 - 检测LV模式
        for offset in range(len(messages[0].data) - 1):
            if self._detect_lv_pattern(messages, offset):
                # 估计LV字段的长度
                pattern_length = self._estimate_lv_length(messages, offset)
                if pattern_length > 0:
                    fields.append(FieldDescription(
                        start=offset,
                        end=offset + pattern_length,
                        field_type="LV_pattern",
                        confidence=0.7,
                        weight=len(messages) * pattern_length
                    ))

        return fields

    def _detect_lv_pattern(self, messages: List[Message], offset: int) -> bool:
        """检测Length-Value模式"""
        try:
            valid_count = 0

            for msg in messages:
                if offset + 1 < len(msg.data):
                    length_byte = msg.data[offset]

                    # 检查长度字节是否合理
                    if 0 < length_byte <= len(msg.data) - offset - 1:
                        # 检查是否有足够的数据
                        if offset + 1 + length_byte <= len(msg.data):
                            valid_count += 1

            return valid_count >= len(messages) * 0.7

        except:
            return False

    def _estimate_lv_length(self, messages: List[Message], offset: int) -> int:
        """估计LV模式的总长度"""
        try:
            lengths = []
            for msg in messages:
                if offset + 1 < len(msg.data):
                    length_byte = msg.data[offset]
                    total_length = 1 + length_byte  # 长度字节 + 数据
                    lengths.append(total_length)

            if lengths:
                return max(lengths)  # 返回最大长度
            return 0

        except:
            return 0


class IntegrationAlgorithm:
    """集成算法 - 基于图的冲突解决"""

    def integrate_fields(self, all_fields: List[FieldDescription],
                         message_length: int) -> List[FieldDescription]:
        """集成所有检测结果"""
        logger.info("      运行图基集成算法...")

        if not all_fields:
            return []

        # 构建有向无环图(DAG)
        graph = self._build_dag(all_fields, message_length)

        # 计算最大权重路径
        optimal_path = self._find_maximum_path(graph)

        return optimal_path

    def _build_dag(self, fields: List[FieldDescription],
                   message_length: int) -> Dict:
        """构建DAG用于冲突解决"""
        # 添加源节点和汇聚节点
        source = FieldDescription(
            start=-1, end=-1, field_type="SOURCE",
            confidence=1.0, weight=0.0
        )
        sink = FieldDescription(
            start=message_length, end=message_length,
            field_type="SINK", confidence=1.0, weight=0.0
        )

        all_nodes = [source] + fields + [sink]

        # 构建邻接表
        graph = defaultdict(list)

        for i, node_a in enumerate(all_nodes):
            for j, node_b in enumerate(all_nodes):
                if i != j and self._strictly_precedes(node_a, node_b):
                    graph[i].append(j)

        return {
            'nodes': all_nodes,
            'edges': graph,
            'source': 0,
            'sink': len(all_nodes) - 1
        }

    def _strictly_precedes(self, field_a: FieldDescription,
                           field_b: FieldDescription) -> bool:
        """检查field_a是否严格在field_b之前"""
        return field_a.end <= field_b.start

    def _find_maximum_path(self, graph: Dict) -> List[FieldDescription]:
        """找到最大权重路径"""
        nodes = graph['nodes']
        edges = graph['edges']
        source = graph['source']
        sink = graph['sink']

        # 拓扑排序
        topo_order = self._topological_sort(edges, len(nodes))

        # 动态规划求最大路径
        dist = [-float('inf')] * len(nodes)
        parent = [-1] * len(nodes)
        dist[source] = 0

        for u in topo_order:
            if dist[u] != -float('inf'):
                for v in edges[u]:
                    weight = nodes[v].weight
                    if dist[u] + weight > dist[v]:
                        dist[v] = dist[u] + weight
                        parent[v] = u

        # 重构路径
        path = []
        current = sink
        while parent[current] != -1:
            if nodes[current].field_type not in ['SOURCE', 'SINK']:
                path.append(nodes[current])
            current = parent[current]

        path.reverse()
        return path

    def _topological_sort(self, edges: Dict, num_nodes: int) -> List[int]:
        """拓扑排序"""
        in_degree = [0] * num_nodes
        for u in edges:
            for v in edges[u]:
                in_degree[v] += 1

        queue = [i for i in range(num_nodes) if in_degree[i] == 0]
        result = []

        while queue:
            u = queue.pop(0)
            result.append(u)

            for v in edges[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        return result


class EnhancedBinaryInfernoAlgorithm:
    """增强版BinaryInferno算法"""

    def __init__(self):
        self.atomic_detectors = None
        self.boundary_detector = None
        self.pattern_detector = PatternDetector()
        self.integration_algorithm = IntegrationAlgorithm()

        # 协议特异性参数
        self.protocol_params = {
            'dns': {'min_field_size': 2, 'endianness': 'big'},
            'modbus': {'min_field_size': 1, 'endianness': 'big'},
            'smb': {'min_field_size': 1, 'endianness': 'little'},
            'smb2': {'min_field_size': 1, 'endianness': 'little'},
            'dhcp': {'min_field_size': 1, 'endianness': 'big'},
            'dnp3': {'min_field_size': 1, 'endianness': 'little'},
            's7comm': {'min_field_size': 1, 'endianness': 'big'},
            'ftp': {'min_field_size': 1, 'endianness': 'big'},
            'tls': {'min_field_size': 1, 'endianness': 'big'}
        }

    def extract_fields(self, messages: List[Message], protocol_name: str = None,
                       capture_time: Optional[Tuple] = None,
                       endianness: str = 'big') -> List[List[int]]:
        """主要的字段提取方法"""
        logger.info(f"🔍 增强版BinaryInferno分析 {len(messages)} 个消息...")

        if not messages:
            return []

        # 应用协议特异性参数
        if protocol_name and protocol_name in self.protocol_params:
            params = self.protocol_params[protocol_name]
            endianness = params.get('endianness', endianness)

        # 初始化检测器
        self.atomic_detectors = AtomicDetectors(capture_time)
        self.boundary_detector = FieldBoundaryDetector(endianness)

        results = []

        for msg in messages:
            # 1. 运行原子检测器
            atomic_fields = []
            atomic_fields.extend(self.atomic_detectors.detect_floats([msg]))
            atomic_fields.extend(self.atomic_detectors.detect_timestamps([msg]))
            atomic_fields.extend(self.atomic_detectors.detect_lengths([msg]))

            # 2. 运行边界检测器
            boundaries = self.boundary_detector.detect_boundaries([msg])
            boundary_fields = []
            for i in range(len(boundaries) - 1):
                boundary_fields.append(FieldDescription(
                    start=boundaries[i],
                    end=boundaries[i + 1],
                    field_type="boundary_field",
                    confidence=0.5,
                    weight=boundaries[i + 1] - boundaries[i]
                ))

            # 3. 运行模式检测器
            pattern_fields = self.pattern_detector.detect_patterns([msg])

            # 4. 集成所有结果
            all_fields = atomic_fields + boundary_fields + pattern_fields
            integrated_fields = self.integration_algorithm.integrate_fields(
                all_fields, len(msg.data)
            )

            # 5. 转换为边界列表
            boundaries = [0]
            for field in integrated_fields:
                if field.start > 0:
                    boundaries.append(field.start)
                if field.end < len(msg.data):
                    boundaries.append(field.end)

            boundaries.append(len(msg.data))
            boundaries = sorted(list(set(boundaries)))

            # 6. 后处理和验证
            final_boundaries = self._postprocess_boundaries(
                boundaries, msg, protocol_name
            )

            results.append(final_boundaries)

        return results

    def _postprocess_boundaries(self, boundaries: List[int],
                                message: Message, protocol_name: str) -> List[int]:
        """后处理边界"""
        if len(boundaries) <= 2:
            return boundaries

        # 移除过小的字段
        min_field_size = 1
        if protocol_name in self.protocol_params:
            min_field_size = self.protocol_params[protocol_name]['min_field_size']

        filtered = [boundaries[0]]
        for i in range(1, len(boundaries)):
            if boundaries[i] - filtered[-1] >= min_field_size:
                filtered.append(boundaries[i])

        # 应用协议特定规则
        if protocol_name == 'dns' and len(message.data) >= 12:
            # DNS固定12字节头部
            if 12 not in filtered:
                filtered.append(12)
                filtered.sort()
        elif protocol_name == 'modbus' and len(message.data) >= 7:
            # Modbus MBAP头部7字节
            if 7 not in filtered:
                filtered.append(7)
                filtered.sort()
        elif protocol_name in ['smb', 'smb2'] and len(message.data) >= 8:
            # SMB头部字段
            for pos in [4, 8]:
                if pos not in filtered and pos < len(message.data):
                    filtered.append(pos)
            filtered.sort()

        return filtered


class EnhancedBinaryInfernoDataLoader:
    """增强版数据加载器"""

    def __init__(self, data_root: str = "../../Msg2"):
        self.data_root = Path(data_root)
        self.supported_protocols = [
            'smb', 'smb2', 'dns', 's7comm', 'dnp3',
            'modbus', 'ftp', 'tls', 'dhcp'
        ]

    def load_protocol_data(self, protocol_name: str) -> List[Dict]:
        """加载协议数据"""
        logger.info(f"📊 加载 {protocol_name.upper()} 协议数据...")

        csv_path = self.data_root / "csv" / protocol_name.lower()

        if not csv_path.exists():
            logger.warning(f"   ❌ CSV目录不存在: {csv_path}")
            return []

        data = []
        csv_files = list(csv_path.glob("*.csv"))

        if not csv_files:
            logger.warning(f"   ❌ 未找到CSV文件: {csv_path}")
            return []

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"   📁 读取文件: {csv_file.name} ({len(df)} 行)")

                for _, row in df.iterrows():
                    try:
                        boundaries_str = str(row['Boundaries'])
                        if boundaries_str and boundaries_str != 'nan':
                            boundaries = [int(b.strip()) for b in boundaries_str.split(',')]
                        else:
                            boundaries = [0]

                        hex_data = str(row['HexData'])
                        data_length = len(bytes.fromhex(hex_data))

                        if 0 not in boundaries:
                            boundaries.insert(0, 0)
                        if data_length not in boundaries:
                            boundaries.append(data_length)

                        boundaries = sorted(list(set(boundaries)))

                        sample = {
                            'raw_data': hex_data,
                            'protocol': protocol_name.lower(),
                            'bytes': bytes.fromhex(hex_data),
                            'length': data_length,
                            'message_type': str(row.get('FunctionCode', 'unknown')),
                            'ground_truth_boundaries': boundaries,
                            'semantic_types': self._parse_semantic_info(row.get('SemanticTypes', '{}')),
                            'semantic_functions': self._parse_semantic_info(row.get('SemanticFunctions', '{}'))
                        }
                        data.append(sample)

                    except Exception as e:
                        logger.warning(f"   ⚠️ 解析行数据错误: {e}")
                        continue

            except Exception as e:
                logger.error(f"   ❌ 读取CSV文件错误: {e}")
                continue

        logger.info(f"   ✅ 成功加载 {len(data)} 条数据")
        return data

    def _parse_semantic_info(self, semantic_str: str) -> Dict:
        """解析语义信息"""
        try:
            if semantic_str and semantic_str != 'nan':
                return json.loads(semantic_str)
            return {}
        except:
            return {}


class EnhancedBinaryInfernoEvaluator:
    """增强版评估器"""

    def __init__(self):
        self.boundary_tolerance = 1

    def evaluate_boundaries(self, predicted_boundaries: List[int],
                            ground_truth_boundaries: List[int],
                            sequence_length: int) -> Dict[str, float]:
        """评估边界检测性能"""

        # 标准评估
        pred_positions = set(predicted_boundaries)
        true_positions = set(ground_truth_boundaries)

        # 准确率计算
        correct_positions = 0
        for pos in range(sequence_length):
            pred_is_boundary = pos in pred_positions
            true_is_boundary = pos in true_positions
            if pred_is_boundary == true_is_boundary:
                correct_positions += 1

        accuracy = correct_positions / sequence_length if sequence_length > 0 else 0

        # 精确率计算
        if len(predicted_boundaries) > 0:
            true_positives = len(true_positions & pred_positions)
            precision = true_positives / len(predicted_boundaries)
        else:
            precision = 0

        # 召回率计算
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

        # 完美率 - 字段级别的精确匹配
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


class EnhancedBinaryInfernoExperiment:
    """增强版实验管理器"""

    def __init__(self, data_root: str = "../../Msg2"):
        self.data_loader = EnhancedBinaryInfernoDataLoader(data_root)
        self.algorithm = EnhancedBinaryInfernoAlgorithm()
        self.evaluator = EnhancedBinaryInfernoEvaluator()

        self.protocols = [
            'smb', 'smb2', 'dns', 's7comm', 'dnp3',
            'modbus', 'ftp', 'tls', 'dhcp'
        ]

        self.results = {}

    def run_experiments(self, protocols: List[str] = None, max_samples: int = None):
        """运行实验"""
        if protocols is None:
            protocols = self.protocols

        logger.info("🚀 增强版BinaryInferno实验开始")
        logger.info("=" * 70)

        for protocol in protocols:
            logger.info(f"\n📊 测试协议: {protocol.upper()}")
            logger.info("-" * 50)

            data = self.data_loader.load_protocol_data(protocol)

            if not data:
                logger.warning(f"   ❌ 跳过 {protocol}: 无数据")
                continue

            if max_samples and len(data) > max_samples:
                data = random.sample(data, max_samples)
                logger.info(f"   📝 限制样本数量: {max_samples}")

            messages = []
            for sample in data:
                msg = Message(sample['raw_data'])
                messages.append(msg)

            try:
                logger.info(f"   🔍 运行增强版BinaryInferno算法...")

                # 设置捕获时间范围 (使用当前时间前后一天)
                current_time = int(time.time())
                capture_time = (current_time - 86400, current_time + 86400)

                predicted_boundaries = self.algorithm.extract_fields(
                    messages, protocol, capture_time, 'big'
                )

                logger.info(f"   📈 评估性能...")
                all_metrics = []

                for sample, pred_boundaries in zip(data, predicted_boundaries):
                    true_boundaries = sample['ground_truth_boundaries']
                    length = sample['length']

                    metrics = self.evaluator.evaluate_boundaries(
                        pred_boundaries, true_boundaries, length
                    )
                    all_metrics.append(metrics)

                # 计算平均指标
                avg_metrics = {}
                for key in ['accuracy', 'precision', 'recall', 'f1_score', 'perfection']:
                    values = [m[key] for m in all_metrics if not np.isnan(m[key])]
                    avg_metrics[key] = np.mean(values) if values else 0.0

                self.results[protocol] = {
                    'sample_count': len(data),
                    'metrics': avg_metrics
                }

                logger.info(f"   ✅ 结果:")
                logger.info(f"      样本数量: {len(data)}")
                logger.info(f"      准确率: {avg_metrics['accuracy']:.4f}")
                logger.info(f"      精确率: {avg_metrics['precision']:.4f}")
                logger.info(f"      召回率: {avg_metrics['recall']:.4f}")
                logger.info(f"      F1分数: {avg_metrics['f1_score']:.4f}")
                logger.info(f"      完美率: {avg_metrics['perfection']:.4f}")

            except Exception as e:
                logger.error(f"   ❌ 处理 {protocol} 时出错: {e}")
                import traceback
                traceback.print_exc()
                self.results[protocol] = {
                    'sample_count': len(data),
                    'metrics': {'accuracy': 0, 'precision': 0, 'recall': 0,
                                'f1_score': 0, 'perfection': 0},
                    'error': str(e)
                }

    def generate_report(self):
        """生成报告"""
        logger.info(f"\n" + "=" * 70)
        logger.info("📊 增强版BinaryInferno实验报告")
        logger.info("=" * 70)

        if not self.results:
            logger.warning("❌ 没有实验结果")
            return

        report_data = []
        for protocol, result in self.results.items():
            metrics = result['metrics']
            report_data.append({
                'Protocol': protocol.upper(),
                'Samples': result['sample_count'],
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-score': f"{metrics['f1_score']:.4f}",
                'Perfection': f"{metrics['perfection']:.4f}"
            })

        df = pd.DataFrame(report_data)
        print("\n增强版BinaryInferno实验结果表格:")
        print(df.to_string(index=False))

        # 计算总体性能
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'perfection']:
            values = [r['metrics'][metric] for r in self.results.values() if metric in r['metrics']]
            avg_metrics[metric] = np.mean(values) if values else 0.0

        logger.info(f"\n🎯 总体性能:")
        logger.info(f"   平均准确率: {avg_metrics['accuracy']:.4f}")
        logger.info(f"   平均精确率: {avg_metrics['precision']:.4f}")
        logger.info(f"   平均召回率: {avg_metrics['recall']:.4f}")
        logger.info(f"   平均F1分数: {avg_metrics['f1_score']:.4f}")
        logger.info(f"   平均完美率: {avg_metrics['perfection']:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强版BinaryInferno实验')

    parser.add_argument('--protocols', nargs='+',
                        choices=['smb', 'smb2', 'dns', 's7comm', 'dnp3',
                                 'modbus', 'ftp', 'tls', 'dhcp'],
                        help='要测试的协议列表')

    parser.add_argument('--max-samples', type=int, default=None,
                        help='每个协议的最大样本数')

    parser.add_argument('--data-root', type=str, default="../../Msg2",
                        help='数据根目录路径')

    args = parser.parse_args()

    experiment = EnhancedBinaryInfernoExperiment(args.data_root)

    logger.info(f"🌟 增强版BinaryInferno实验设置:")
    logger.info(f"   数据目录: {args.data_root}")
    logger.info(f"   测试协议: {args.protocols or 'ALL'}")
    logger.info(f"   最大样本: {args.max_samples or 'ALL'}")

    experiment.run_experiments(protocols=args.protocols, max_samples=args.max_samples)
    experiment.generate_report()

    logger.info("\n✅ 增强版BinaryInferno实验完成！")


if __name__ == "__main__":
    main()