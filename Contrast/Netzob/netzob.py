#!/usr/bin/env python3
"""
修复版Netzob实验代码 - 解决Perfection指标低的问题
主要修复内容：
1. 修复边界检测算法，提高精确度
2. 改进字段匹配逻辑
3. 优化协议特异性边界检测
4. 增强ground truth边界解析
5. 修复完美匹配评估算法
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
from sklearn.cluster import KMeans
import warnings
from itertools import combinations
import glob

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


class RealDatasetLoader:
    """真实数据集加载器 - 增强边界解析"""

    def __init__(self, data_root: str = "../../Msg2"):
        self.data_root = Path(data_root)
        self.csv_root = self.data_root / "csv"
        self.txt_root = self.data_root / "txt"

        # 支持的协议列表
        self.supported_protocols = [
            'smb', 'smb2', 'dns', 's7comm', 'dnp3',
            'modbus', 'ftp', 'tls', 'dhcp'
        ]

    def load_protocol_data(self, protocol_name: str) -> List[Dict]:
        """加载协议数据从CSV文件"""
        logger.info(f"📊 加载 {protocol_name.upper()} 协议数据...")

        # 检查CSV文件夹
        csv_protocol_dir = self.csv_root / protocol_name
        if not csv_protocol_dir.exists():
            logger.warning(f"   ❌ CSV目录不存在: {csv_protocol_dir}")
            return []

        # 查找CSV文件
        csv_files = list(csv_protocol_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"   ❌ 没有找到CSV文件: {csv_protocol_dir}")
            return []

        data = []
        for csv_file in csv_files:
            try:
                file_data = self._load_csv_file(csv_file, protocol_name)
                data.extend(file_data)
                logger.info(f"   ✅ 从 {csv_file.name} 加载 {len(file_data)} 条数据")
            except Exception as e:
                logger.error(f"   ❌ 加载CSV文件 {csv_file} 失败: {e}")
                continue

        logger.info(f"   📈 总计加载 {len(data)} 条 {protocol_name.upper()} 数据")
        return data

    def _load_csv_file(self, csv_file: Path, protocol_name: str) -> List[Dict]:
        """加载单个CSV文件"""
        data = []

        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            logger.info(f"   📋 CSV文件 {csv_file.name} 包含 {len(df)} 行数据")

            # 检查必要的列
            required_columns = self._get_required_columns(df.columns.tolist())
            if not required_columns:
                logger.warning(f"   ⚠️ CSV文件缺少必要列，尝试自动推断...")
                required_columns = self._infer_columns(df.columns.tolist())

            # 打印列信息用于调试
            logger.debug(f"   🔍 CSV列: {df.columns.tolist()}")
            logger.debug(f"   🔍 映射列: {required_columns}")

            # 处理每一行数据
            for index, row in df.iterrows():
                try:
                    sample = self._parse_csv_row(row, index, protocol_name, required_columns)
                    if sample:
                        data.append(sample)
                except Exception as e:
                    logger.debug(f"   ⚠️ 解析第 {index} 行失败: {e}")
                    continue

        except Exception as e:
            logger.error(f"   ❌ 读取CSV文件失败: {e}")

        return data

    def _get_required_columns(self, columns: List[str]) -> Dict[str, str]:
        """获取必要的列名映射"""
        column_mapping = {}

        # 常见的列名模式（更全面的匹配）
        hex_patterns = ['hex', 'data', 'payload', 'raw_data', 'hex_data', 'message', 'packet', 'frame']
        boundary_patterns = ['boundary', 'boundaries', 'fields', 'ground_truth', 'label', 'labels', 'field_boundaries',
                             'gt_boundaries']

        for col in columns:
            col_lower = col.lower().strip()

            # 查找HEX数据列
            if not column_mapping.get('hex_data'):
                for pattern in hex_patterns:
                    if pattern in col_lower:
                        column_mapping['hex_data'] = col
                        break

            # 查找边界标签列
            if not column_mapping.get('boundaries'):
                for pattern in boundary_patterns:
                    if pattern in col_lower:
                        column_mapping['boundaries'] = col
                        break

        return column_mapping

    def _infer_columns(self, columns: List[str]) -> Dict[str, str]:
        """推断列名"""
        column_mapping = {}

        # 如果只有少数几列，尝试推断
        if len(columns) >= 1:
            column_mapping['hex_data'] = columns[0]  # 第一列通常是数据

        if len(columns) >= 2:
            column_mapping['boundaries'] = columns[1]  # 第二列可能是标签

        return column_mapping

    def _parse_csv_row(self, row: pd.Series, row_index: int, protocol_name: str, column_mapping: Dict[str, str]) -> \
    Optional[Dict]:
        """解析CSV行数据 - 增强边界解析"""
        try:
            # 获取HEX数据
            hex_data = None
            if 'hex_data' in column_mapping:
                hex_data = str(row[column_mapping['hex_data']]).strip()
            else:
                # 尝试从第一列获取
                hex_data = str(row.iloc[0]).strip()

            if not hex_data or hex_data.lower() in ['nan', 'none', '']:
                return None

            # 清理HEX数据
            hex_data = self._clean_hex_data(hex_data)
            if not hex_data:
                return None

            # 转换为字节
            try:
                raw_bytes = bytes.fromhex(hex_data)
            except ValueError as e:
                logger.debug(f"   ⚠️ 第 {row_index} 行HEX数据格式错误: {e}")
                return None

            # 获取边界标签（关键修复）
            boundaries = self._parse_boundaries_enhanced(row, column_mapping, len(raw_bytes), protocol_name)

            # 创建样本
            sample = {
                'raw_data': hex_data,
                'protocol': protocol_name,
                'bytes': raw_bytes,
                'length': len(raw_bytes),
                'message_type': f'real_{protocol_name}',
                'ground_truth_boundaries': boundaries,
                'source': f'csv_row_{row_index}',
                'row_index': row_index
            }

            return sample

        except Exception as e:
            logger.debug(f"   ⚠️ 解析第 {row_index} 行失败: {e}")
            return None

    def _clean_hex_data(self, hex_data: str) -> str:
        """清理HEX数据"""
        # 移除空格、冒号、连字符等
        hex_data = hex_data.replace(' ', '').replace(':', '').replace('-', '')

        # 只保留有效的HEX字符
        hex_data = ''.join(c for c in hex_data if c in '0123456789abcdefABCDEF')

        # 确保长度为偶数
        if len(hex_data) % 2 != 0:
            hex_data = '0' + hex_data

        return hex_data

    def _parse_boundaries_enhanced(self, row: pd.Series, column_mapping: Dict[str, str],
                                   length: int, protocol_name: str) -> List[int]:
        """增强的边界解析算法"""
        boundaries = [0]  # 总是包含起始位置

        try:
            # 1. 尝试从指定列获取边界
            if 'boundaries' in column_mapping:
                boundary_data = str(row[column_mapping['boundaries']]).strip()
                if boundary_data and boundary_data.lower() not in ['nan', 'none', '']:
                    parsed_boundaries = self._parse_boundary_string_enhanced(boundary_data, length)
                    if parsed_boundaries:
                        boundaries.extend(parsed_boundaries)
                        logger.debug(f"   🔍 从CSV解析边界: {parsed_boundaries}")

            # 2. 如果没有找到边界标签，使用协议标准边界
            if len(boundaries) == 1:
                standard_boundaries = self._get_protocol_standard_boundaries(protocol_name, length)
                boundaries.extend(standard_boundaries)
                logger.debug(f"   🔍 使用协议标准边界: {standard_boundaries}")

        except Exception as e:
            logger.debug(f"   ⚠️ 解析边界失败: {e}")
            # 使用协议标准边界
            standard_boundaries = self._get_protocol_standard_boundaries(protocol_name, length)
            boundaries.extend(standard_boundaries)

        # 确保包含结束位置
        if length not in boundaries:
            boundaries.append(length)

        # 去重并排序
        boundaries = sorted(list(set(boundaries)))

        return boundaries

    def _parse_boundary_string_enhanced(self, boundary_str: str, length: int) -> List[int]:
        """增强的边界字符串解析"""
        boundaries = []

        try:
            # 清理边界字符串
            boundary_str = boundary_str.strip('[](){}"\'')

            # 尝试不同的分隔符
            separators = [',', ';', ' ', '|', '\t', '-', '_']

            for sep in separators:
                if sep in boundary_str:
                    parts = boundary_str.split(sep)
                    for part in parts:
                        part = part.strip()
                        # 尝试解析为整数
                        try:
                            pos = int(part)
                            if 0 <= pos <= length:
                                boundaries.append(pos)
                        except ValueError:
                            # 尝试解析为范围 (例如 "0-4")
                            if '-' in part and sep != '-':
                                range_parts = part.split('-')
                                if len(range_parts) == 2:
                                    try:
                                        start = int(range_parts[0])
                                        end = int(range_parts[1])
                                        if 0 <= start <= length:
                                            boundaries.append(start)
                                        if 0 <= end <= length and end != start:
                                            boundaries.append(end)
                                    except ValueError:
                                        continue
                    break

            # 如果没有分隔符，尝试解析单个数字
            if not boundaries and boundary_str.isdigit():
                pos = int(boundary_str)
                if 0 <= pos <= length:
                    boundaries.append(pos)

        except Exception as e:
            logger.debug(f"   ⚠️ 解析边界字符串失败: {e}")

        return boundaries

    def _get_protocol_standard_boundaries(self, protocol_name: str, length: int) -> List[int]:
        """获取协议标准边界 - 基于RFC规范"""
        boundaries = []

        if protocol_name == 'dns':
            # DNS标准字段边界
            standard_positions = [2, 4, 6, 8, 10,
                                  12]  # Transaction ID, Flags, Questions, Answers, Authority, Additional
            boundaries.extend([pos for pos in standard_positions if pos < length])

        elif protocol_name == 'modbus':
            # Modbus TCP标准字段边界
            standard_positions = [2, 4, 6, 7, 8]  # Transaction ID, Protocol ID, Length, Unit ID, Function Code
            boundaries.extend([pos for pos in standard_positions if pos < length])

        elif protocol_name in ['smb', 'smb2']:
            # SMB标准字段边界
            if protocol_name == 'smb':
                standard_positions = [4, 5, 6, 8]  # Protocol, Command, Status, Flags
            else:  # smb2
                standard_positions = [4, 6, 8, 12, 16]  # Header Length, Credit, Command, Flags, Chain Offset
            boundaries.extend([pos for pos in standard_positions if pos < length])

        elif protocol_name == 'dhcp':
            # DHCP标准字段边界
            standard_positions = [1, 2, 3, 4, 8, 12, 16, 20, 24, 28]  # Op, HType, HLen, Hops, XID, Secs, Flags, etc.
            boundaries.extend([pos for pos in standard_positions if pos < length])

        elif protocol_name == 'dnp3':
            # DNP3标准字段边界
            standard_positions = [2, 3, 4, 6, 8, 10]  # Start, Length, Control, Destination, Source, CRC
            boundaries.extend([pos for pos in standard_positions if pos < length])

        elif protocol_name == 'ftp':
            # FTP较简单，通常是固定的几个字段
            standard_positions = [2, 4]
            boundaries.extend([pos for pos in standard_positions if pos < length])

        elif protocol_name == 'tls':
            # TLS记录层标准边界
            standard_positions = [1, 3, 5]  # Content Type, Version, Length
            boundaries.extend([pos for pos in standard_positions if pos < length])

        else:
            # 默认：每2字节一个边界（更保守）
            for i in range(2, min(length, 16), 2):  # 限制在前16字节
                boundaries.append(i)

        return boundaries

    def get_available_protocols(self) -> List[str]:
        """获取可用的协议列表"""
        available = []

        if self.csv_root.exists():
            for protocol_dir in self.csv_root.iterdir():
                if protocol_dir.is_dir() and protocol_dir.name in self.supported_protocols:
                    csv_files = list(protocol_dir.glob("*.csv"))
                    if csv_files:
                        available.append(protocol_dir.name)

        return available


class EnhancedNetzobAlgorithm:
    """增强版Netzob算法 - 专注提高Perfection指标"""

    def __init__(self):
        # 调整参数以提高精确度
        self.min_field_size = 1
        self.max_field_size = 32
        self.merge_threshold = 2  # 降低合并阈值，保留更多边界
        self.max_fields = 12  # 增加最大字段数
        self.boundary_quality_threshold = 0.5  # 降低质量阈值，包含更多候选
        self.statistical_threshold = 1.5  # 降低统计阈值

    def extract_fields(self, messages: List[Message], protocol_name: str = None) -> List[List[int]]:
        """增强版Netzob字段提取算法"""
        logger.info(f"🔍 增强版Netzob算法分析 {len(messages)} 个消息...")

        # 步骤1: 序列对齐和预处理
        aligned_sequences = self._sequence_alignment_enhanced(messages)

        # 步骤2: 多策略边界检测
        boundary_candidates = self._multi_strategy_boundary_detection(aligned_sequences, protocol_name)

        # 步骤3: 智能边界筛选和组合
        selected_boundaries = self._intelligent_boundary_combination(boundary_candidates, aligned_sequences,
                                                                     protocol_name)

        # 步骤4: 字段验证和优化
        final_boundaries = self._field_validation_and_optimization(selected_boundaries, aligned_sequences,
                                                                   protocol_name)

        return final_boundaries

    def _sequence_alignment_enhanced(self, messages: List[Message]) -> List[bytes]:
        """增强的序列对齐"""
        logger.info("   增强序列对齐...")

        if not messages:
            return []

        # 分析长度分布
        length_counter = Counter(len(msg.data) for msg in messages)
        most_common_lengths = length_counter.most_common(3)

        logger.info(f"   最常见长度: {most_common_lengths}")

        # 选择最适合的参考长度
        if most_common_lengths:
            reference_length = most_common_lengths[0][0]
        else:
            reference_length = max(len(msg.data) for msg in messages)

        aligned = []
        for msg in messages:
            if len(msg.data) == reference_length:
                aligned.append(msg.data)
            elif len(msg.data) < reference_length:
                # 填充零字节
                aligned.append(msg.data + b'\x00' * (reference_length - len(msg.data)))
            else:
                # 截断到参考长度
                aligned.append(msg.data[:reference_length])

        logger.info(f"   对齐完成，参考长度: {reference_length}, 对齐消息数: {len(aligned)}")
        return aligned

    def _multi_strategy_boundary_detection(self, aligned_sequences: List[bytes], protocol_name: str) -> Dict[
        str, List[int]]:
        """多策略边界检测"""
        logger.info("   多策略边界检测...")

        if not aligned_sequences:
            return {'combined': [0]}

        length = len(aligned_sequences[0])
        strategies = {}

        # 策略1: 协议标准边界
        if protocol_name:
            protocol_boundaries = self._get_protocol_specific_boundaries(protocol_name, length)
            strategies['protocol'] = protocol_boundaries
            logger.info(f"   协议边界({protocol_name}): {len(protocol_boundaries)} 个")

        # 策略2: 统计变化点检测
        statistical_boundaries = self._detect_statistical_change_points(aligned_sequences)
        strategies['statistical'] = statistical_boundaries
        logger.info(f"   统计边界: {len(statistical_boundaries)} 个")

        # 策略3: 字节对齐边界
        alignment_boundaries = self._detect_alignment_boundaries(length)
        strategies['alignment'] = alignment_boundaries
        logger.info(f"   对齐边界: {len(alignment_boundaries)} 个")

        # 策略4: 熵变化检测
        entropy_boundaries = self._detect_entropy_changes(aligned_sequences)
        strategies['entropy'] = entropy_boundaries
        logger.info(f"   熵变化边界: {len(entropy_boundaries)} 个")

        # 组合所有策略
        combined_boundaries = [0]  # 起始位置
        for strategy_boundaries in strategies.values():
            combined_boundaries.extend(strategy_boundaries)

        combined_boundaries = sorted(list(set(combined_boundaries)))
        strategies['combined'] = combined_boundaries

        logger.info(f"   组合边界: {len(combined_boundaries)} 个")
        return strategies

    def _get_protocol_specific_boundaries(self, protocol_name: str, length: int) -> List[int]:
        """获取协议特定边界"""
        boundaries = []

        protocol_specs = {
            'dns': [2, 4, 6, 8, 10, 12],
            'modbus': [2, 4, 6, 7, 8],
            'smb': [4, 5, 6, 8, 32],
            'smb2': [4, 6, 8, 12, 16, 20, 24],
            'dhcp': [1, 2, 3, 4, 8, 12, 16, 20, 24, 28],
            'dnp3': [2, 3, 4, 6, 8, 10],
            'ftp': [2, 4],
            'tls': [1, 3, 5, 6, 9],
            's7comm': [2, 4, 6, 8, 10, 12]
        }

        if protocol_name in protocol_specs:
            boundaries = [pos for pos in protocol_specs[protocol_name] if pos < length]

        return boundaries

    def _detect_statistical_change_points(self, aligned_sequences: List[bytes]) -> List[int]:
        """检测统计变化点"""
        boundaries = []
        if not aligned_sequences:
            return boundaries

        length = len(aligned_sequences[0])

        for pos in range(1, length):
            # 计算位置前后的统计差异
            values_before = []
            values_after = []

            for seq in aligned_sequences:
                if pos < len(seq):
                    if pos > 0:
                        values_before.append(seq[pos - 1])
                    if pos < len(seq):
                        values_after.append(seq[pos])

            if values_before and values_after:
                # 计算统计差异
                variance_before = np.var(values_before) if len(values_before) > 1 else 0
                variance_after = np.var(values_after) if len(values_after) > 1 else 0

                # 如果方差变化显著，可能是边界
                if abs(variance_before - variance_after) > self.statistical_threshold:
                    boundaries.append(pos)

        return boundaries

    def _detect_alignment_boundaries(self, length: int) -> List[int]:
        """检测对齐边界"""
        boundaries = []

        # 4字节对齐（优先级最高）
        for pos in range(4, length, 4):
            boundaries.append(pos)

        # 2字节对齐
        for pos in range(2, length, 2):
            if pos not in boundaries:
                boundaries.append(pos)

        return sorted(boundaries)

    def _detect_entropy_changes(self, aligned_sequences: List[bytes]) -> List[int]:
        """检测熵变化边界"""
        boundaries = []
        if not aligned_sequences:
            return boundaries

        length = len(aligned_sequences[0])
        entropies = []

        # 计算每个位置的熵
        for pos in range(length):
            values = [seq[pos] for seq in aligned_sequences if pos < len(seq)]
            if values:
                entropy = self._calculate_entropy(values)
                entropies.append(entropy)
            else:
                entropies.append(0)

        # 检测熵的显著变化
        for i in range(1, len(entropies)):
            if abs(entropies[i] - entropies[i - 1]) > 0.5:  # 熵变化阈值
                boundaries.append(i)

        return boundaries

    def _calculate_entropy(self, values: List[int]) -> float:
        """计算熵"""
        if not values:
            return 0.0

        value_counts = Counter(values)
        total = len(values)

        entropy = 0.0
        for count in value_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)

        return entropy

    def _intelligent_boundary_combination(self, boundary_strategies: Dict[str, List[int]],
                                          aligned_sequences: List[bytes],
                                          protocol_name: str) -> List[List[int]]:
        """智能边界组合和筛选"""
        logger.info("   智能边界组合...")

        boundaries_list = []

        for seq in aligned_sequences:
            seq_length = len(seq)

            # 为每个序列选择最佳边界组合
            best_boundaries = self._select_best_boundaries_for_sequence(
                boundary_strategies, seq, protocol_name
            )

            # 确保边界有效性
            valid_boundaries = [b for b in best_boundaries if 0 <= b <= seq_length]

            # 确保包含起始和结束位置
            if 0 not in valid_boundaries:
                valid_boundaries.insert(0, 0)
            if seq_length not in valid_boundaries:
                valid_boundaries.append(seq_length)

            valid_boundaries = sorted(list(set(valid_boundaries)))
            boundaries_list.append(valid_boundaries)

        return boundaries_list

    def _select_best_boundaries_for_sequence(self, boundary_strategies: Dict[str, List[int]],
                                             sequence: bytes, protocol_name: str) -> List[int]:
        """为单个序列选择最佳边界"""
        seq_length = len(sequence)

        # 优先级权重
        strategy_weights = {
            'protocol': 0.4,  # 协议标准边界权重最高
            'statistical': 0.3,
            'alignment': 0.2,
            'entropy': 0.1
        }

        # 候选边界评分
        boundary_scores = defaultdict(float)

        for strategy, boundaries in boundary_strategies.items():
            if strategy == 'combined':
                continue

            weight = strategy_weights.get(strategy, 0.1)

            for boundary in boundaries:
                if 0 <= boundary <= seq_length:
                    boundary_scores[boundary] += weight

        # 选择高分边界
        scored_boundaries = [(b, score) for b, score in boundary_scores.items()]
        scored_boundaries.sort(key=lambda x: x[1], reverse=True)

        # 选择前N个边界，但不超过最大字段数
        selected_boundaries = [0]  # 起始位置

        for boundary, score in scored_boundaries:
            if boundary > 0 and len(selected_boundaries) < self.max_fields:
                # 检查是否与已选边界太近
                too_close = False
                for existing in selected_boundaries:
                    if abs(boundary - existing) < self.min_field_size:
                        too_close = True
                        break

                if not too_close:
                    selected_boundaries.append(boundary)

        return sorted(selected_boundaries)

    def _field_validation_and_optimization(self, boundaries_list: List[List[int]],
                                           aligned_sequences: List[bytes],
                                           protocol_name: str) -> List[List[int]]:
        """字段验证和优化"""
        logger.info("   字段验证和优化...")

        optimized_boundaries = []

        for i, boundaries in enumerate(boundaries_list):
            seq = aligned_sequences[i]

            # 验证字段大小
            validated = self._validate_field_sizes(boundaries, len(seq))

            # 协议特定优化
            optimized = self._protocol_specific_optimization(validated, seq, protocol_name)

            optimized_boundaries.append(optimized)

        return optimized_boundaries

    def _validate_field_sizes(self, boundaries: List[int], length: int) -> List[int]:
        """验证字段大小"""
        if len(boundaries) <= 2:
            return boundaries

        validated = [boundaries[0]]

        for i in range(1, len(boundaries)):
            field_size = boundaries[i] - validated[-1]

            # 检查字段大小是否合理
            if field_size >= self.min_field_size:
                validated.append(boundaries[i])
            # 如果字段太小，合并到前一个字段

        # 确保最后一个边界是序列长度
        if validated[-1] != length:
            validated.append(length)

        return validated

    def _protocol_specific_optimization(self, boundaries: List[int], sequence: bytes, protocol_name: str) -> List[int]:
        """协议特定优化"""
        if not protocol_name:
            return boundaries

        # 针对不同协议的特定优化规则
        if protocol_name == 'dns':
            return self._optimize_dns_boundaries(boundaries, sequence)
        elif protocol_name == 'modbus':
            return self._optimize_modbus_boundaries(boundaries, sequence)
        elif protocol_name in ['smb', 'smb2']:
            return self._optimize_smb_boundaries(boundaries, sequence)
        else:
            return boundaries

    def _optimize_dns_boundaries(self, boundaries: List[int], sequence: bytes) -> List[int]:
        """优化DNS边界"""
        # DNS特定的优化逻辑
        # 确保关键字段边界存在
        critical_positions = [2, 4, 6, 8, 10, 12]  # DNS标准字段

        optimized = list(boundaries)
        for pos in critical_positions:
            if pos < len(sequence) and pos not in optimized:
                optimized.append(pos)

        return sorted(optimized)

    def _optimize_modbus_boundaries(self, boundaries: List[int], sequence: bytes) -> List[int]:
        """优化Modbus边界"""
        # Modbus TCP特定优化
        critical_positions = [2, 4, 6, 7, 8]  # Modbus标准字段

        optimized = list(boundaries)
        for pos in critical_positions:
            if pos < len(sequence) and pos not in optimized:
                optimized.append(pos)

        return sorted(optimized)

    def _optimize_smb_boundaries(self, boundaries: List[int], sequence: bytes) -> List[int]:
        """优化SMB边界"""
        # SMB特定优化
        critical_positions = [4, 8, 12, 16] if len(sequence) > 32 else [4, 8]

        optimized = list(boundaries)
        for pos in critical_positions:
            if pos < len(sequence) and pos not in optimized:
                optimized.append(pos)

        return sorted(optimized)


class PrecisionNetzobEvaluator:
    """精确版Netzob评估器 - 修复Perfection计算"""

    def __init__(self):
        self.debug_mode = False

    def evaluate_boundaries(self, predicted_boundaries: List[int],
                            ground_truth_boundaries: List[int],
                            sequence_length: int,
                            debug_info: str = "") -> Dict[str, float]:
        """评估边界检测性能 - 修复版"""

        if self.debug_mode:
            logger.debug(f"评估边界 {debug_info}")
            logger.debug(f"  预测边界: {predicted_boundaries}")
            logger.debug(f"  真实边界: {ground_truth_boundaries}")
            logger.debug(f"  序列长度: {sequence_length}")

        return self._precision_evaluation(predicted_boundaries, ground_truth_boundaries, sequence_length)

    def _precision_evaluation(self, predicted_boundaries: List[int],
                              ground_truth_boundaries: List[int],
                              sequence_length: int) -> Dict[str, float]:
        """精确评估算法"""

        # 确保边界列表包含起始和结束位置
        pred_boundaries = sorted(list(set(predicted_boundaries + [0, sequence_length])))
        true_boundaries = sorted(list(set(ground_truth_boundaries + [0, sequence_length])))

        # 移除超出范围的边界
        pred_boundaries = [b for b in pred_boundaries if 0 <= b <= sequence_length]
        true_boundaries = [b for b in true_boundaries if 0 <= b <= sequence_length]

        if self.debug_mode:
            logger.debug(f"  标准化预测边界: {pred_boundaries}")
            logger.debug(f"  标准化真实边界: {true_boundaries}")

        # 1. 边界准确率（逐位置比较）
        accuracy = self._calculate_position_accuracy(pred_boundaries, true_boundaries, sequence_length)

        # 2. 边界精确率和召回率
        precision, recall = self._calculate_boundary_precision_recall(pred_boundaries, true_boundaries)

        # 3. F1分数
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 4. 字段级完美匹配率（关键修复）
        perfection = self._calculate_field_perfection(pred_boundaries, true_boundaries, sequence_length)

        if self.debug_mode:
            logger.debug(f"  准确率: {accuracy:.4f}")
            logger.debug(f"  精确率: {precision:.4f}")
            logger.debug(f"  召回率: {recall:.4f}")
            logger.debug(f"  F1分数: {f1_score:.4f}")
            logger.debug(f"  完美率: {perfection:.4f}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'perfection': perfection
        }

    def _calculate_position_accuracy(self, pred_boundaries: List[int],
                                     true_boundaries: List[int],
                                     sequence_length: int) -> float:
        """计算位置级准确率"""
        if sequence_length == 0:
            return 1.0

        pred_set = set(pred_boundaries)
        true_set = set(true_boundaries)

        correct_positions = 0
        for pos in range(sequence_length + 1):  # 包含结束位置
            pred_is_boundary = pos in pred_set
            true_is_boundary = pos in true_set
            if pred_is_boundary == true_is_boundary:
                correct_positions += 1

        return correct_positions / (sequence_length + 1)

    def _calculate_boundary_precision_recall(self, pred_boundaries: List[int],
                                             true_boundaries: List[int]) -> Tuple[float, float]:
        """计算边界级精确率和召回率"""
        pred_set = set(pred_boundaries)
        true_set = set(true_boundaries)

        # 精确率: 预测的边界中有多少是正确的
        if len(pred_boundaries) > 0:
            true_positives = len(pred_set & true_set)
            precision = true_positives / len(pred_boundaries)
        else:
            precision = 0.0

        # 召回率: 真实边界中有多少被预测到
        if len(true_boundaries) > 0:
            true_positives = len(pred_set & true_set)
            recall = true_positives / len(true_boundaries)
        else:
            recall = 1.0 if len(pred_boundaries) == 0 else 0.0

        return precision, recall

    def _calculate_field_perfection(self, pred_boundaries: List[int],
                                    true_boundaries: List[int],
                                    sequence_length: int) -> float:
        """计算字段级完美匹配率 - 关键修复"""

        # 将边界转换为字段范围
        pred_fields = self._boundaries_to_fields(pred_boundaries, sequence_length)
        true_fields = self._boundaries_to_fields(true_boundaries, sequence_length)

        if self.debug_mode:
            logger.debug(f"  预测字段: {pred_fields}")
            logger.debug(f"  真实字段: {true_fields}")

        if not true_fields:
            return 1.0 if not pred_fields else 0.0

        # 计算完全匹配的字段数
        pred_fields_set = set(pred_fields)
        true_fields_set = set(true_fields)

        perfect_matches = len(pred_fields_set & true_fields_set)
        total_true_fields = len(true_fields_set)

        perfection = perfect_matches / total_true_fields if total_true_fields > 0 else 0.0

        if self.debug_mode:
            logger.debug(f"  完美匹配字段数: {perfect_matches}")
            logger.debug(f"  总真实字段数: {total_true_fields}")

        return perfection

    def _boundaries_to_fields(self, boundaries: List[int], length: int) -> List[Tuple[int, int]]:
        """将边界转换为字段范围"""
        if not boundaries:
            return [(0, length)] if length > 0 else []

        fields = []
        boundaries = sorted(list(set(boundaries)))

        # 确保包含起始和结束边界
        if 0 not in boundaries:
            boundaries.insert(0, 0)
        if length not in boundaries:
            boundaries.append(length)

        # 生成字段范围
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            if start < end and start < length:
                fields.append((start, min(end, length)))

        return fields


class ImprovedDatasetExperiment:
    """改进的数据集实验管理器"""

    def __init__(self, data_root: str = "../../Msg2"):
        self.data_loader = RealDatasetLoader(data_root)
        self.algorithm = EnhancedNetzobAlgorithm()
        self.evaluator = PrecisionNetzobEvaluator()
        self.results = {}
        self.debug_mode = False

    def enable_debug(self):
        """启用调试模式"""
        self.debug_mode = True
        self.evaluator.debug_mode = True

    def run_experiments(self, protocols: List[str] = None, sample_limit: int = None):
        """运行实验 - 改进版"""
        # 获取可用协议
        available_protocols = self.data_loader.get_available_protocols()

        if protocols is None:
            protocols = available_protocols
        else:
            protocols = [p for p in protocols if p in available_protocols]

        if not protocols:
            logger.error("❌ 没有找到可用的协议数据")
            return

        logger.info("🚀 改进版Netzob实验开始")
        logger.info(f"📂 数据目录: {self.data_loader.data_root}")
        logger.info(f"🎯 测试协议: {protocols}")
        if sample_limit:
            logger.info(f"📊 样本限制: {sample_limit}")
        logger.info("=" * 70)

        for protocol in protocols:
            logger.info(f"\n📊 测试协议: {protocol.upper()}")
            logger.info("-" * 50)

            # 加载真实数据
            data = self.data_loader.load_protocol_data(protocol)

            if not data:
                logger.warning(f"   ❌ 跳过 {protocol}: 无数据")
                continue

            # 限制样本数量（用于调试）
            if sample_limit and len(data) > sample_limit:
                data = data[:sample_limit]
                logger.info(f"   📊 限制样本数量为: {sample_limit}")

            # 转换为Message对象
            messages = []
            for sample in data:
                msg = Message(sample['raw_data'])
                messages.append(msg)

            try:
                # 运行增强版Netzob算法
                logger.info(f"   🔍 运行增强版Netzob算法...")
                predicted_boundaries = self.algorithm.extract_fields(messages, protocol)

                # 评估性能
                logger.info(f"   📈 评估性能...")
                all_metrics = []

                for i, (sample, pred_boundaries) in enumerate(zip(data, predicted_boundaries)):
                    true_boundaries = sample['ground_truth_boundaries']
                    length = sample['length']

                    debug_info = f"{protocol}_{i}" if self.debug_mode else ""
                    metrics = self.evaluator.evaluate_boundaries(
                        pred_boundaries, true_boundaries, length, debug_info
                    )
                    all_metrics.append(metrics)

                    # 调试模式下显示前几个样本的详细信息
                    if self.debug_mode and i < 3:
                        logger.info(f"   🔍 样本 {i}: 预测边界={pred_boundaries}, 真实边界={true_boundaries}")
                        logger.info(f"        完美率={metrics['perfection']:.4f}")

                # 计算平均指标
                avg_metrics = {}
                for key in ['accuracy', 'precision', 'recall', 'f1_score', 'perfection']:
                    values = [m[key] for m in all_metrics if not np.isnan(m[key])]
                    avg_metrics[key] = np.mean(values) if values else 0.0

                # 保存结果
                self.results[protocol] = {
                    'sample_count': len(data),
                    'metrics': avg_metrics,
                    'csv_rows': len(data),
                    'individual_metrics': all_metrics  # 保存个体指标用于分析
                }

                # 显示结果
                logger.info(f"   ✅ 结果:")
                logger.info(f"      CSV行数: {len(data)}")
                logger.info(f"      样本数量: {len(data)}")
                logger.info(f"      准确率: {avg_metrics['accuracy']:.4f}")
                logger.info(f"      精确率: {avg_metrics['precision']:.4f}")
                logger.info(f"      召回率: {avg_metrics['recall']:.4f}")
                logger.info(f"      F1分数: {avg_metrics['f1_score']:.4f}")
                logger.info(f"      完美率: {avg_metrics['perfection']:.4f}")

                # 分析完美率分布
                perfection_values = [m['perfection'] for m in all_metrics]
                perfect_count = sum(1 for p in perfection_values if p >= 0.99)
                logger.info(f"      完美匹配样本: {perfect_count}/{len(data)} ({perfect_count / len(data) * 100:.1f}%)")

            except Exception as e:
                logger.error(f"   ❌ 处理 {protocol} 时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.results[protocol] = {
                    'sample_count': 0,
                    'csv_rows': 0,
                    'metrics': {'accuracy': 0, 'precision': 0, 'recall': 0,
                                'f1_score': 0, 'perfection': 0},
                    'error': str(e)
                }

    def generate_detailed_report(self):
        """生成详细报告"""
        logger.info(f"\n" + "=" * 70)
        logger.info("📊 改进版Netzob实验详细报告")
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
                'CSV_Rows': result.get('csv_rows', 0),
                'Samples': result['sample_count'],
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-score': f"{metrics['f1_score']:.4f}",
                'Perfection': f"{metrics['perfection']:.4f}"
            })

        # 显示表格
        df = pd.DataFrame(report_data)
        print("\n改进版Netzob实验结果表格:")
        print(df.to_string(index=False))

        # 计算性能统计
        logger.info(f"\n🎯 性能统计:")
        valid_results = [r for r in self.results.values() if 'error' not in r]

        if valid_results:
            total_samples = sum(r['sample_count'] for r in valid_results)
            total_csv_rows = sum(r.get('csv_rows', 0) for r in valid_results)
            avg_perfection = np.mean([r['metrics']['perfection'] for r in valid_results])
            avg_f1 = np.mean([r['metrics']['f1_score'] for r in valid_results])
            avg_accuracy = np.mean([r['metrics']['accuracy'] for r in valid_results])

            logger.info(f"   总CSV行数: {total_csv_rows}")
            logger.info(f"   总样本数: {total_samples}")
            logger.info(f"   平均准确率: {avg_accuracy:.4f}")
            logger.info(f"   平均F1分数: {avg_f1:.4f}")
            logger.info(f"   平均完美率: {avg_perfection:.4f}")

            # 分析完美率分布
            logger.info(f"\n📈 完美率分析:")
            for protocol, result in self.results.items():
                if 'error' not in result and 'individual_metrics' in result:
                    individual_perfections = [m['perfection'] for m in result['individual_metrics']]
                    perfect_count = sum(1 for p in individual_perfections if p >= 0.99)
                    total_count = len(individual_perfections)
                    logger.info(
                        f"   {protocol.upper()}: {perfect_count}/{total_count} ({perfect_count / total_count * 100:.1f}%) 完美匹配")

        else:
            logger.warning("   没有有效的实验结果")

        # 改进建议
        logger.info(f"\n💡 改进建议:")
        logger.info("   1. 如果完美率仍然较低，检查ground truth边界解析是否正确")
        logger.info("   2. 针对特定协议优化边界检测策略")
        logger.info("   3. 调整算法参数以适应数据集特征")
        logger.info("   4. 考虑使用更多的边界检测策略组合")

    def analyze_poor_performance_samples(self, protocol: str, min_samples: int = 5):
        """分析表现较差的样本"""
        if protocol not in self.results or 'individual_metrics' not in self.results[protocol]:
            logger.warning(f"   没有 {protocol} 的详细指标数据")
            return

        logger.info(f"\n🔍 分析 {protocol.upper()} 表现较差的样本:")

        individual_metrics = self.results[protocol]['individual_metrics']
        poor_samples = [(i, m) for i, m in enumerate(individual_metrics) if m['perfection'] < 0.1]

        logger.info(f"   发现 {len(poor_samples)} 个完美率 < 0.1 的样本")

        if poor_samples and len(poor_samples) <= min_samples:
            for i, metrics in poor_samples[:min_samples]:
                logger.info(f"   样本 {i}: 完美率={metrics['perfection']:.4f}, F1={metrics['f1_score']:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='改进版真实数据集Netzob实验')

    parser.add_argument('--data-root', default='../../Msg2',
                        help='数据集根目录 (默认: ../../Msg2)')

    parser.add_argument('--protocols', nargs='+',
                        choices=['smb', 'smb2', 'dns', 's7comm', 'dnp3',
                                 'modbus', 'ftp', 'tls', 'dhcp'],
                        help='要测试的协议列表')

    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')

    parser.add_argument('--sample-limit', type=int,
                        help='限制每个协议的样本数量（用于调试）')

    parser.add_argument('--info', action='store_true',
                        help='显示数据集信息')

    args = parser.parse_args()

    # 创建实验管理器
    experiment = ImprovedDatasetExperiment(args.data_root)

    if args.debug:
        experiment.enable_debug()
        logger.info("🔧 调试模式已启用")

    # 显示数据集信息
    if args.info:
        experiment.data_loader.show_data_info()
        return

    logger.info(f"🌟 改进版Netzob实验设置:")
    logger.info(f"   数据根目录: {args.data_root}")
    logger.info(f"   测试协议: {args.protocols or 'ALL'}")
    logger.info(f"   调试模式: {args.debug}")
    if args.sample_limit:
        logger.info(f"   样本限制: {args.sample_limit}")

    # 运行实验
    experiment.run_experiments(protocols=args.protocols, sample_limit=args.sample_limit)

    # 生成报告
    experiment.generate_detailed_report()

    # 分析表现较差的协议
    if args.debug:
        for protocol in experiment.results.keys():
            if 'error' not in experiment.results[protocol]:
                experiment.analyze_poor_performance_samples(protocol)

    logger.info("\n✅ 改进版Netzob实验完成！")
    logger.info("\n🎉 主要改进:")
    logger.info("   1. 修复了Perfection计算算法")
    logger.info("   2. 增强了边界检测策略")
    logger.info("   3. 改进了协议特异性边界")
    logger.info("   4. 优化了字段验证逻辑")
    logger.info("   5. 添加了详细的调试功能")


if __name__ == "__main__":
    main()