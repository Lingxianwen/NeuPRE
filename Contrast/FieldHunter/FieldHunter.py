#!/usr/bin/env python3
"""
改进的FieldHunter - 修复Perfection问题
主要改进：
1. 去掉Tolerant_Perfection指标
2. 增加调试信息分析Perfection低的原因
3. 改进边界检测和评估策略
4. 增加详细的分析报告
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Union
from collections import defaultdict, Counter
import json
import random
from pathlib import Path
import argparse
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cluster import DBSCAN
import warnings
from dataclasses import dataclass
import math
import glob

warnings.filterwarnings('ignore')


@dataclass
class FieldCandidate:
    """字段候选结构"""
    start_offset: int
    end_offset: int
    field_type: str
    confidence: float
    pattern_consistency: float

    @property
    def length(self) -> int:
        return self.end_offset - self.start_offset

    def overlaps_with(self, other: 'FieldCandidate') -> bool:
        """检查是否与另一个字段候选重叠"""
        return not (self.end_offset <= other.start_offset or self.start_offset >= other.end_offset)


def count_protocol_files(data_root: str = "../../Msg2", protocol_name: str = None) -> int:
    """统计指定协议的数据文件数量"""
    data_path = Path(data_root)

    if not data_path.exists():
        print(f"⚠️  警告: 数据根目录 '{data_root}' 不存在，使用默认样本数量")
        return 100

    if protocol_name:
        # 统计特定协议的数据行数
        csv_protocol_path = data_path / "csv" / protocol_name
        if csv_protocol_path.exists():
            total_rows = 0
            csv_files = list(csv_protocol_path.glob("*.csv"))
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    total_rows += len(df)
                except Exception as e:
                    print(f"⚠️  读取文件失败 {csv_file}: {e}")
            print(f"📁 协议 {protocol_name.upper()} 数据条数: {total_rows}")
            return total_rows
        else:
            print(f"⚠️  警告: 协议目录 '{csv_protocol_path}' 不存在")
            return 0
    else:
        # 统计所有协议的平均数据行数
        csv_path = data_path / "csv"
        if not csv_path.exists():
            print(f"⚠️  警告: csv目录 '{csv_path}' 不存在")
            return 100

        total_rows = 0
        protocol_count = 0

        for protocol_dir in csv_path.iterdir():
            if protocol_dir.is_dir():
                protocol_rows = 0
                csv_files = list(protocol_dir.glob("*.csv"))
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        protocol_rows += len(df)
                    except Exception as e:
                        print(f"⚠️  读取文件失败 {csv_file}: {e}")

                total_rows += protocol_rows
                protocol_count += 1
                print(f"📁 协议 {protocol_dir.name.upper()}: {protocol_rows} 条数据")

        if protocol_count > 0:
            avg_rows = total_rows // protocol_count
            print(f"📊 平均每个协议数据条数: {avg_rows}")
            return avg_rows
        else:
            return 100


class RealDataFieldHunterDataLoader:
    """使用真实数据集的FieldHunter数据加载器"""

    def __init__(self, data_root: str = "../../Msg2"):
        self.data_root = Path(data_root)
        self.txt_path = self.data_root / "txt"
        self.csv_path = self.data_root / "csv"

        # 检查数据目录结构
        self._validate_data_structure()

        # 获取支持的协议列表
        self.supported_protocols = self._get_available_protocols()

    def _validate_data_structure(self):
        """验证数据目录结构"""
        if not self.data_root.exists():
            raise FileNotFoundError(f"数据根目录不存在: {self.data_root}")

        if not self.csv_path.exists():
            raise FileNotFoundError(f"csv目录不存在: {self.csv_path}")

        print(f"✅ 数据目录结构验证通过: {self.data_root}")

    def _get_available_protocols(self) -> List[str]:
        """获取可用的协议列表"""
        protocols = []
        for protocol_dir in self.csv_path.iterdir():
            if protocol_dir.is_dir():
                protocols.append(protocol_dir.name.lower())

        print(f"📋 发现协议: {protocols}")
        return sorted(protocols)

    def load_protocol_data(self, protocol_name: str) -> List[Dict]:
        """加载指定协议的真实数据"""
        protocol_name = protocol_name.lower()

        if protocol_name not in self.supported_protocols:
            print(f"❌ 不支持的协议: {protocol_name}")
            print(f"   支持的协议: {self.supported_protocols}")
            return []

        print(f"📊 加载 {protocol_name.upper()} 协议真实数据...")

        # 直接从CSV文件加载数据（新格式）
        data = self._load_csv_data_new_format(protocol_name)

        print(f"   ✅ 成功加载 {len(data)} 条真实数据")
        return data

    def _load_csv_data_new_format(self, protocol_name: str) -> List[Dict]:
        """加载新格式的CSV数据（每行包含HexData和Boundaries）"""
        protocol_csv_path = self.csv_path / protocol_name
        data = []

        if not protocol_csv_path.exists():
            print(f"   ⚠️  协议csv目录不存在: {protocol_csv_path}")
            return data

        csv_files = list(protocol_csv_path.glob("*.csv"))
        print(f"   📊 找到 {len(csv_files)} 个数据文件")

        for csv_file in csv_files:
            try:
                # 读取CSV文件
                df = pd.read_csv(csv_file)
                print(f"   📄 处理文件: {csv_file.name} ({len(df)} 条记录)")

                # 处理每一行数据
                for idx, row in df.iterrows():
                    try:
                        # 获取HEX数据
                        hex_data = str(row.get('HexData', ''))
                        if not hex_data:
                            continue

                        # 清理HEX数据
                        hex_data = self._clean_hex_data(hex_data)
                        if not hex_data:
                            continue

                        # 转换为bytes
                        raw_bytes = bytes.fromhex(hex_data)

                        # 获取边界信息
                        boundaries_str = str(row.get('Boundaries', ''))
                        boundaries = self._parse_boundaries_string(boundaries_str, len(raw_bytes))

                        # 创建数据记录
                        sample = {
                            'file_id': f"{csv_file.stem}_{idx}",
                            'raw_data': hex_data,
                            'protocol': protocol_name,
                            'bytes': raw_bytes,
                            'length': len(raw_bytes),
                            'ground_truth_boundaries': boundaries,
                            'function_code': row.get('FunctionCode', ''),
                            'has_boundary': row.get('HasBoundary', False),
                            'boundary_count': row.get('BoundaryCount', 0),
                            'semantic_type': row.get('SemanticType', ''),
                            'label': row.get('Label', '')
                        }
                        data.append(sample)

                    except Exception as e:
                        print(f"   ⚠️  处理第{idx}行数据失败: {e}")
                        continue

            except Exception as e:
                print(f"   ⚠️  读取文件失败 {csv_file.name}: {e}")
                continue

        return data

    def _parse_boundaries_string(self, boundaries_str: str, max_length: int) -> List[int]:
        """解析边界字符串"""
        boundaries = [0]  # 始终包含起始位置

        if not boundaries_str or boundaries_str == 'nan':
            return boundaries

        try:
            # 解析逗号分隔的边界字符串
            boundary_parts = boundaries_str.split(',')
            for part in boundary_parts:
                boundary = int(part.strip())
                if 0 <= boundary <= max_length:
                    boundaries.append(boundary)
        except (ValueError, AttributeError) as e:
            print(f"   ⚠️  解析边界字符串失败: {boundaries_str}, 错误: {e}")

        # 确保包含结束位置
        if max_length not in boundaries:
            boundaries.append(max_length)

        return sorted(list(set(boundaries)))

    def _clean_hex_data(self, hex_content: str) -> str:
        """清理HEX数据"""
        # 移除空格、换行符等
        cleaned = ''.join(hex_content.split())

        # 移除非HEX字符
        cleaned = ''.join(c for c in cleaned if c.lower() in '0123456789abcdef')

        # 确保是偶数长度
        if len(cleaned) % 2 != 0:
            cleaned = cleaned[:-1]  # 移除最后一个字符

        return cleaned


class ImprovedFieldHunterAlgorithm:
    """改进的FieldHunter算法 - 参考NetPlier成功策略"""

    def __init__(self, min_field_size: int = 1, max_field_size: int = 64):
        self.min_field_size = min_field_size
        self.max_field_size = max_field_size

        # 参考NetPlier的参数
        self.merge_threshold = 2
        self.confidence_threshold = 0.6  # 提高置信度阈值
        self.boundary_tolerance = 1

        # 协议特异性参数
        self.protocol_params = {
            'dns': {'min_field_size': 2, 'merge_threshold': 1},
            'modbus': {'min_field_size': 1, 'merge_threshold': 2},
            'smb': {'min_field_size': 1, 'merge_threshold': 4},
            'dhcp': {'min_field_size': 1, 'merge_threshold': 3}
        }

    def extract_fields(self, packet_data: List[bytes], protocol_name: str = None) -> List[List[int]]:
        """从数据包列表中提取字段边界 - 改进版"""
        if not packet_data:
            return []

        print(f"🔍 改进版FieldHunter分析 {len(packet_data)} 个数据包...")

        # 应用协议特异性参数
        if protocol_name and protocol_name in self.protocol_params:
            params = self.protocol_params[protocol_name]
            self.min_field_size = params.get('min_field_size', self.min_field_size)
            self.merge_threshold = params.get('merge_threshold', self.merge_threshold)

        # 统一包长度分析
        lengths = [len(packet) for packet in packet_data]
        common_length = self._find_common_length(lengths)

        if common_length:
            print(f"   检测到常见包长度: {common_length}")
            same_length_packets = [p for p in packet_data if len(p) == common_length]
            if len(same_length_packets) >= max(10, len(packet_data) * 0.3):
                return self._analyze_fixed_length_packets_improved(same_length_packets, packet_data, protocol_name)

        # 变长包分析
        return self._analyze_variable_length_packets_improved(packet_data, protocol_name)

    def _find_common_length(self, lengths: List[int]) -> Optional[int]:
        """找到最常见的包长度"""
        length_counts = Counter(lengths)
        if not length_counts:
            return None

        most_common_length, count = length_counts.most_common(1)[0]
        if count >= max(10, len(lengths) * 0.3):
            return most_common_length

        return None

    def _analyze_fixed_length_packets_improved(self, same_length_packets: List[bytes],
                                               all_packets: List[bytes],
                                               protocol_name: str) -> List[List[int]]:
        """改进的固定长度数据包分析"""
        if not same_length_packets:
            return [[] for _ in all_packets]

        packet_length = len(same_length_packets[0])
        print(f"   分析固定长度包 (长度={packet_length})")

        # 1. 检测高质量边界候选 - 增强版
        high_quality_candidates = self._detect_comprehensive_boundaries(same_length_packets, protocol_name)

        # 2. 智能边界选择 - 更宽松的策略
        selected_boundaries = self._enhanced_boundary_selection(high_quality_candidates, packet_length)

        # 3. NetPlier风格后处理 - 减少过度合并
        final_boundaries = self._flexible_postprocessing(selected_boundaries, packet_length, protocol_name)

        print(f"   检测到边界: {final_boundaries}")

        # 为所有包应用边界模式
        result = []
        for packet in all_packets:
            if len(packet) == packet_length:
                result.append(final_boundaries)
            else:
                adjusted_boundaries = self._adjust_boundaries_for_length(
                    final_boundaries, len(packet), packet_length
                )
                result.append(adjusted_boundaries)

        return result

    def _detect_comprehensive_boundaries(self, packets: List[bytes], protocol_name: str) -> List[int]:
        """检测全面的边界候选 - 增强版"""
        print("   检测全面边界候选...")

        if not packets:
            return [0]

        packet_length = len(packets[0])
        candidates = [0]  # 始终包含起始位置

        # 1. 基于熵变化的边界检测
        entropy_boundaries = self._detect_entropy_change_boundaries(packets)
        candidates.extend(entropy_boundaries)

        # 2. 基于字节值分布变化的边界检测
        distribution_boundaries = self._detect_distribution_change_boundaries(packets)
        candidates.extend(distribution_boundaries)

        # 3. 基于相关性变化的边界检测
        correlation_boundaries = self._detect_correlation_boundaries(packets)
        candidates.extend(correlation_boundaries)

        # 4. 协议特异性边界
        if protocol_name:
            protocol_boundaries = self._detect_protocol_specific_boundaries(packets, protocol_name)
            candidates.extend(protocol_boundaries)

        # 5. 对齐边界
        alignment_boundaries = self._detect_alignment_boundaries(packet_length)
        candidates.extend(alignment_boundaries)

        # 去重并排序
        candidates = sorted(list(set(candidates)))

        # 过滤低质量边界 - 使用更宽松的阈值
        quality_candidates = self._filter_quality_boundaries_relaxed(candidates, packets)

        print(f"   检测到 {len(quality_candidates)} 个边界候选")
        return quality_candidates

    def _detect_entropy_change_boundaries(self, packets: List[bytes]) -> List[int]:
        """基于熵变化的边界检测"""
        boundaries = []
        if not packets:
            return boundaries

        packet_length = len(packets[0])

        # 计算每个位置的熵
        entropies = []
        for pos in range(packet_length):
            values = [packet[pos] for packet in packets if pos < len(packet)]
            if values:
                entropy = self._calculate_entropy(values)
                entropies.append(entropy)
            else:
                entropies.append(0)

        # 寻找熵的显著变化点
        for i in range(1, len(entropies) - 1):
            left_entropy = entropies[i - 1]
            curr_entropy = entropies[i]
            right_entropy = entropies[i + 1]

            # 如果当前位置的熵与邻近位置有显著差异
            if abs(curr_entropy - left_entropy) > 0.3 or abs(curr_entropy - right_entropy) > 0.3:
                boundaries.append(i)

        return boundaries

    def _detect_distribution_change_boundaries(self, packets: List[bytes]) -> List[int]:
        """基于字节值分布变化的边界检测"""
        boundaries = []
        if not packets:
            return boundaries

        packet_length = len(packets[0])

        # 使用滑动窗口检测分布变化
        window_size = 3
        for pos in range(window_size, packet_length - window_size):
            # 获取窗口内的字节值分布
            left_values = []
            right_values = []

            for packet in packets:
                if pos < len(packet):
                    # 左窗口
                    for i in range(max(0, pos - window_size), pos):
                        if i < len(packet):
                            left_values.append(packet[i])
                    # 右窗口
                    for i in range(pos, min(len(packet), pos + window_size)):
                        right_values.append(packet[i])

            if left_values and right_values:
                # 计算分布的差异
                left_dist = self._calculate_distribution_stats(left_values)
                right_dist = self._calculate_distribution_stats(right_values)

                # 如果分布有显著差异，认为是边界
                if self._distributions_differ(left_dist, right_dist):
                    boundaries.append(pos)

        return boundaries

    def _detect_correlation_boundaries(self, packets: List[bytes]) -> List[int]:
        """基于相关性变化的边界检测"""
        boundaries = []
        if not packets or len(packets[0]) < 4:
            return boundaries

        packet_length = len(packets[0])

        # 计算相邻字节的相关性
        for pos in range(1, packet_length - 1):
            correlations = []

            for i in range(len(packets)):
                if pos + 1 < len(packets[i]):
                    # 计算当前位置与下一位置的相关性
                    curr_byte = packets[i][pos]
                    next_byte = packets[i][pos + 1]
                    correlations.append(abs(curr_byte - next_byte))

            if correlations:
                avg_correlation = np.mean(correlations)
                std_correlation = np.std(correlations)

                # 如果相关性变化很大，可能是边界
                if std_correlation > 30:  # 调整阈值
                    boundaries.append(pos + 1)

        return boundaries

    def _detect_protocol_specific_boundaries(self, packets: List[bytes], protocol_name: str) -> List[int]:
        """检测协议特异性边界"""
        boundaries = []
        if not packets:
            return boundaries

        packet_length = len(packets[0])

        if protocol_name == 'dns':
            # DNS协议的固定头部是12字节
            if packet_length >= 12:
                boundaries.extend([2, 4, 6, 8, 10, 12])
        elif protocol_name == 'modbus':
            # Modbus协议的MBAP头部是7字节
            if packet_length >= 7:
                boundaries.extend([2, 4, 6, 7])
            # 添加PDU相关边界
            if packet_length >= 9:
                boundaries.extend([8, 9])
        elif protocol_name in ['smb', 'smb2']:
            # SMB协议的头部字段
            if packet_length >= 8:
                boundaries.extend([4, 8])
            if packet_length >= 16:
                boundaries.extend([12, 16])
        elif protocol_name == 'dhcp':
            # DHCP协议的常见字段边界
            if packet_length >= 28:
                boundaries.extend([1, 2, 3, 4, 8, 12, 16, 20, 24, 28])

        return [b for b in boundaries if b < packet_length]

    def _detect_alignment_boundaries(self, packet_length: int) -> List[int]:
        """检测对齐边界"""
        boundaries = []

        # 1字节对齐 - 每个位置都可能是边界
        for pos in range(1, packet_length):
            boundaries.append(pos)

        return boundaries

    def _calculate_entropy(self, values: List[int]) -> float:
        """计算熵"""
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

    def _calculate_distribution_stats(self, values: List[int]) -> Dict:
        """计算分布统计"""
        if not values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    def _distributions_differ(self, dist1: Dict, dist2: Dict) -> bool:
        """判断两个分布是否有显著差异"""
        mean_diff = abs(dist1['mean'] - dist2['mean'])
        std_diff = abs(dist1['std'] - dist2['std'])

        return mean_diff > 20 or std_diff > 15

    def _filter_quality_boundaries_relaxed(self, candidates: List[int], packets: List[bytes]) -> List[int]:
        """过滤低质量边界 - 更宽松的策略"""
        if not candidates or not packets:
            return candidates

        # 降低质量阈值，保留更多边界
        quality_threshold = 0.1  # 从0.4降低到0.1
        quality_candidates = []

        for candidate in candidates:
            # 计算该位置的质量分数
            quality_score = self._calculate_boundary_quality_score(candidate, packets)

            # 保留质量分数较高的边界
            if quality_score > quality_threshold:
                quality_candidates.append(candidate)

        return quality_candidates

    def _calculate_boundary_quality_score(self, position: int, packets: List[bytes]) -> float:
        """计算边界质量分数"""
        if not packets or position >= len(packets[0]):
            return 0.0

        score = 0.1  # 基础分数

        # 1. 位置分数（所有位置都有基本分数）
        score += 0.2

        # 2. 对齐分数
        if position % 4 == 0:
            score += 0.3
        elif position % 2 == 0:
            score += 0.2

        # 3. 统计变化分数
        if position > 0 and position < len(packets[0]):
            left_values = [packet[position - 1] for packet in packets if position - 1 < len(packet)]
            right_values = [packet[position] for packet in packets if position < len(packet)]

            if left_values and right_values:
                left_entropy = self._calculate_entropy(left_values)
                right_entropy = self._calculate_entropy(right_values)

                entropy_diff = abs(right_entropy - left_entropy)
                if entropy_diff > 0.1:  # 降低阈值
                    score += 0.2

        return min(1.0, score)

    def _enhanced_boundary_selection(self, candidates: List[int], packet_length: int) -> List[int]:
        """增强的边界选择 - 更宽松的策略"""
        print("   增强边界选择...")

        if not candidates:
            return [0]

        # 按重要性排序候选边界
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_boundary_importance_relaxed(candidate, packet_length)
            scored_candidates.append((candidate, score))

        # 按分数排序
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # 选择更多的高分边界
        selected = [0]  # 始终包含起始位置
        for candidate, score in scored_candidates:
            if score > 0.2 and candidate not in selected:  # 降低阈值从0.5到0.2
                selected.append(candidate)
                # 增加边界数量限制，允许更多分割
                if len(selected) >= min(16, packet_length // 2):  # 从8增加到16
                    break

        return sorted(selected)

    def _calculate_boundary_importance_relaxed(self, position: int, packet_length: int) -> float:
        """计算边界重要性 - 更宽松的策略"""
        if position >= packet_length:
            return 0.0

        importance = 0.1  # 基础重要性

        # 1. 位置重要性 - 所有位置都有一定重要性
        if position < packet_length // 2:  # 前半部分
            importance += 0.3
        else:
            importance += 0.2

        # 2. 对齐重要性
        if position % 4 == 0:
            importance += 0.3
        elif position % 2 == 0:
            importance += 0.2
        else:
            importance += 0.1

        # 3. 相对位置重要性
        relative_pos = position / packet_length
        if relative_pos < 0.8:  # 大部分位置都重要
            importance += 0.2

        return min(1.0, importance)

    def _flexible_postprocessing(self, boundaries: List[int], packet_length: int, protocol_name: str) -> List[int]:
        """灵活的后处理 - 减少过度合并"""
        print("   灵活后处理...")

        # 1. 轻微合并过小的字段
        merged = self._merge_small_fields_flexible(boundaries)

        # 2. 边界对齐
        aligned = self._align_boundaries_flexible(merged, packet_length)

        # 3. 应用协议规则
        if protocol_name:
            aligned = self._apply_protocol_rules(aligned, protocol_name, packet_length)

        # 4. 验证和清理
        final = self._validate_and_clean_boundaries_flexible(aligned, packet_length)

        return final

    def _merge_small_fields_flexible(self, boundaries: List[int]) -> List[int]:
        """灵活合并过小的字段"""
        if len(boundaries) <= 2:
            return boundaries

        merged = [boundaries[0]]

        for i in range(1, len(boundaries)):
            field_size = boundaries[i] - merged[-1]

            # 只合并非常小的字段（1字节），保留更多边界
            if field_size < 1:  # 减少合并阈值
                continue  # 跳过这个边界，实现合并
            else:
                merged.append(boundaries[i])

        return merged

    def _align_boundaries_flexible(self, boundaries: List[int], packet_length: int) -> List[int]:
        """灵活边界对齐"""
        aligned = [boundaries[0]]

        for boundary in boundaries[1:]:
            # 保持原始边界，减少对齐调整
            aligned_boundary = boundary

            # 确保不与前一个边界重复
            if aligned_boundary > aligned[-1]:
                aligned.append(aligned_boundary)

        return aligned

    def _apply_protocol_rules(self, boundaries: List[int], protocol_name: str, packet_length: int) -> List[int]:
        """应用协议规则"""
        if protocol_name == 'dns':
            # DNS协议：确保头部12字节完整
            if 12 not in boundaries and 12 < packet_length:
                boundaries = sorted(boundaries + [12])
        elif protocol_name == 'modbus':
            # Modbus协议：确保MBAP头部7字节
            if 7 not in boundaries and 7 < packet_length:
                boundaries = sorted(boundaries + [7])
        elif protocol_name in ['smb', 'smb2']:
            # SMB协议：确保头部字段
            important_positions = [4, 8]
            for pos in important_positions:
                if pos not in boundaries and pos < packet_length:
                    boundaries = sorted(boundaries + [pos])

        return boundaries

    def _validate_and_clean_boundaries_flexible(self, boundaries: List[int], packet_length: int) -> List[int]:
        """灵活验证和清理边界"""
        # 确保边界在有效范围内
        valid_boundaries = [b for b in boundaries if 0 <= b < packet_length]

        # 确保包含起始位置
        if 0 not in valid_boundaries:
            valid_boundaries.insert(0, 0)

        # 移除重复边界
        valid_boundaries = sorted(list(set(valid_boundaries)))

        # 增加字段数量限制，允许更精细的分割
        max_fields = min(32, packet_length)  # 大幅增加限制
        if len(valid_boundaries) > max_fields:
            valid_boundaries = valid_boundaries[:max_fields]

        return valid_boundaries

    def _analyze_variable_length_packets_improved(self, packet_data: List[bytes], protocol_name: str) -> List[
        List[int]]:
        """改进的变长数据包分析"""
        print(f"   分析变长数据包")

        result = []

        # 按长度分组分析
        length_groups = defaultdict(list)
        for i, packet in enumerate(packet_data):
            length_groups[len(packet)].append((i, packet))

        # 为每个长度组找到边界模式
        length_patterns = {}
        for length, packets in length_groups.items():
            if len(packets) >= 3:  # 降低最小样本要求
                packet_bytes = [p[1] for p in packets]
                boundaries = self._find_boundaries_for_length_group_improved(packet_bytes, protocol_name)
                length_patterns[length] = boundaries

        # 为每个包分配边界
        for i, packet in enumerate(packet_data):
            length = len(packet)
            if length in length_patterns:
                result.append(length_patterns[length])
            else:
                # 使用改进的启发式方法
                boundaries = self._improved_heuristic_boundaries(packet, protocol_name)
                result.append(boundaries)

        return result

    def _find_boundaries_for_length_group_improved(self, packets: List[bytes], protocol_name: str) -> List[int]:
        """为特定长度组找到改进的边界"""
        if not packets:
            return [0]

        # 使用改进的分析方法
        high_quality_candidates = self._detect_comprehensive_boundaries(packets, protocol_name)
        selected_boundaries = self._enhanced_boundary_selection(high_quality_candidates, len(packets[0]))
        final_boundaries = self._flexible_postprocessing(selected_boundaries, len(packets[0]), protocol_name)

        return final_boundaries

    def _improved_heuristic_boundaries(self, packet: bytes, protocol_name: str) -> List[int]:
        """改进的启发式边界方法"""
        length = len(packet)
        boundaries = [0]

        # 协议特异性启发式
        if protocol_name == 'dns' and length >= 12:
            boundaries.extend([2, 4, 6, 8, 10, 12])
        elif protocol_name == 'modbus' and length >= 7:
            boundaries.extend([2, 4, 6, 7])
        elif protocol_name in ['smb', 'smb2'] and length >= 8:
            boundaries.extend([4, 8])
        else:
            # 通用启发式 - 更细粒度的分割
            if length <= 8:
                for i in range(1, length):
                    boundaries.append(i)
            elif length <= 16:
                for i in range(1, length, 1):  # 每个字节都是潜在边界
                    boundaries.append(i)
            elif length <= 32:
                for i in range(2, length, 2):
                    boundaries.append(i)
            else:
                # 长包：多层次分析
                for i in range(4, length, 4):
                    boundaries.append(i)

        # 过滤有效边界
        valid_boundaries = [b for b in boundaries if 0 <= b < length]

        # 应用后处理
        final_boundaries = self._flexible_postprocessing(valid_boundaries, length, protocol_name)

        return final_boundaries

    def _adjust_boundaries_for_length(self, base_boundaries: List[int], target_length: int, base_length: int) -> List[
        int]:
        """为不同长度的包调整边界"""
        if target_length == base_length:
            return base_boundaries

        # 智能比例调整
        ratio = target_length / base_length
        adjusted = []

        for boundary in base_boundaries:
            new_boundary = int(boundary * ratio)

            # 确保边界在有效范围内
            if 0 <= new_boundary < target_length:
                adjusted.append(new_boundary)

        # 确保包含起始位置
        if 0 not in adjusted:
            adjusted.insert(0, 0)

        return sorted(list(set(adjusted)))


class ImprovedFieldHunterEvaluator:
    """改进的FieldHunter评估器 - 增加调试信息"""

    def __init__(self):
        self.boundary_tolerance = 1
        self.debug_mode = True

    def evaluate_boundaries(self, predicted_boundaries: List[int],
                            ground_truth_boundaries: List[int],
                            sequence_length: int,
                            sample_id: str = None) -> Dict[str, float]:
        """评估边界检测性能 - 增加调试信息"""
        if self.debug_mode and sample_id and sample_id.endswith('_0'):  # 只对第一个样本输出详细信息
            print(f"\n   📋 调试信息 (样本 {sample_id}):")
            print(f"      序列长度: {sequence_length}")
            print(f"      真实边界: {ground_truth_boundaries}")
            print(f"      预测边界: {predicted_boundaries}")

            # 转换为字段
            true_fields = self._boundaries_to_fields(ground_truth_boundaries, sequence_length)
            pred_fields = self._boundaries_to_fields(predicted_boundaries, sequence_length)
            print(f"      真实字段: {true_fields}")
            print(f"      预测字段: {pred_fields}")
            print(f"      真实字段数量: {len(true_fields)}")
            print(f"      预测字段数量: {len(pred_fields)}")

        standard_metrics = self._standard_evaluation(predicted_boundaries, ground_truth_boundaries, sequence_length)

        return standard_metrics

    def _standard_evaluation(self, predicted_boundaries: List[int],
                             ground_truth_boundaries: List[int],
                             sequence_length: int) -> Dict[str, float]:
        """标准评估"""
        pred_labels = np.zeros(sequence_length, dtype=int)
        true_labels = np.zeros(sequence_length, dtype=int)

        for boundary in predicted_boundaries:
            if 0 <= boundary < sequence_length:
                pred_labels[boundary] = 1

        for boundary in ground_truth_boundaries:
            if 0 <= boundary < sequence_length:
                true_labels[boundary] = 1

        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)

        tp = np.sum((pred_labels == 1) & (true_labels == 1))
        fp = np.sum((pred_labels == 1) & (true_labels == 0))
        fn = np.sum((pred_labels == 0) & (true_labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

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
            'f1_score': f1,
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


class RealDataFieldHunterExperiment:
    """使用真实数据集的FieldHunter实验管理器"""

    def __init__(self, data_root: str = "../../Msg2"):
        try:
            self.data_loader = RealDataFieldHunterDataLoader(data_root)
            self.algorithm = ImprovedFieldHunterAlgorithm()
            self.evaluator = ImprovedFieldHunterEvaluator()
            self.data_root = data_root

            # 获取支持的协议列表
            self.protocols = self.data_loader.supported_protocols

            if not self.protocols:
                print("❌ 没有找到任何支持的协议数据")

        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            raise

        self.results = {}

    def run_experiments(self, protocols: List[str] = None, max_samples: int = None):
        """运行实验"""
        if protocols is None:
            protocols = self.protocols

        print("🚀 FieldHunter真实数据实验开始")
        print("=" * 70)

        for protocol in protocols:
            if protocol not in self.protocols:
                print(f"❌ 跳过不支持的协议: {protocol}")
                continue

            print(f"\n📊 测试协议: {protocol.upper()}")
            print("-" * 50)

            try:
                data = self.data_loader.load_protocol_data(protocol)

                if not data:
                    print(f"   ❌ 跳过 {protocol}: 无数据")
                    continue

                # 如果没有指定最大样本数，则使用该协议的所有数据
                if max_samples is None:
                    protocol_max_samples = len(data)
                else:
                    protocol_max_samples = max_samples

                if len(data) > protocol_max_samples:
                    data = random.sample(data, protocol_max_samples)
                    print(f"   📝 限制样本数量: {protocol_max_samples}")

                # 分析真实边界统计
                self._analyze_ground_truth_statistics(data, protocol)

                packet_data = [sample['bytes'] for sample in data]

                print(f"   🔍 运行改进版FieldHunter算法...")
                predicted_boundaries = self.algorithm.extract_fields(packet_data, protocol)

                print(f"   📈 评估性能...")
                all_metrics = []

                for sample, pred_boundaries in zip(data, predicted_boundaries):
                    true_boundaries = sample['ground_truth_boundaries']
                    length = sample['length']
                    sample_id = sample['file_id']

                    metrics = self.evaluator.evaluate_boundaries(pred_boundaries, true_boundaries, length, sample_id)
                    all_metrics.append(metrics)

                if all_metrics:
                    avg_metrics = {}
                    for key in ['accuracy', 'precision', 'recall', 'f1_score', 'perfection']:
                        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
                        avg_metrics[key] = np.mean(values) if values else 0.0

                    self.results[protocol] = {
                        'sample_count': len(data),
                        'metrics': avg_metrics
                    }

                    print(f"   ✅ 结果:")
                    print(f"      样本数量: {len(data)}")
                    print(f"      准确率: {avg_metrics['accuracy']:.4f}")
                    print(f"      精确率: {avg_metrics['precision']:.4f}")
                    print(f"      召回率: {avg_metrics['recall']:.4f}")
                    print(f"      F1分数: {avg_metrics['f1_score']:.4f}")
                    print(f"      完美率: {avg_metrics['perfection']:.4f}")
                else:
                    print(f"   ⚠️  评估指标计算失败")

            except Exception as e:
                print(f"   ❌ 处理 {protocol} 时出错: {e}")
                self.results[protocol] = {
                    'sample_count': 0,
                    'metrics': {'accuracy': 0, 'precision': 0, 'recall': 0,
                                'f1_score': 0, 'perfection': 0},
                    'error': str(e)
                }

    def _analyze_ground_truth_statistics(self, data: List[Dict], protocol: str):
        """分析真实边界统计信息"""
        print(f"   📊 分析 {protocol.upper()} 协议真实边界统计:")

        boundary_counts = [len(sample['ground_truth_boundaries']) for sample in data]
        field_counts = [len(sample['ground_truth_boundaries']) - 1 for sample in data if
                        len(sample['ground_truth_boundaries']) > 1]
        packet_lengths = [sample['length'] for sample in data]

        print(f"      数据包长度范围: {min(packet_lengths)} - {max(packet_lengths)}")
        print(f"      平均数据包长度: {np.mean(packet_lengths):.1f}")
        print(f"      平均边界数量: {np.mean(boundary_counts):.1f}")
        print(f"      平均字段数量: {np.mean(field_counts):.1f}" if field_counts else "      平均字段数量: 0")

        # 显示一些示例
        print(f"      边界示例 (前3个样本):")
        for i, sample in enumerate(data[:3]):
            print(f"        样本{i + 1}: 长度={sample['length']}, 边界={sample['ground_truth_boundaries']}")

    def generate_report(self):
        """生成报告"""
        print(f"\n" + "=" * 70)
        print("📊 FieldHunter真实数据实验报告")
        print("=" * 70)

        if not self.results:
            print("❌ 没有实验结果")
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
        print("\nFieldHunter真实数据实验结果表格:")
        print(df.to_string(index=False))

        # 计算平均性能
        valid_results = [r for r in self.results.values() if r['sample_count'] > 0]
        if valid_results:
            avg_perfection = np.mean([r['metrics']['perfection'] for r in valid_results])

            print(f"\n🎯 整体性能:")
            print(f"   平均完美率: {avg_perfection:.4f}")
            print(f"   数据来源: 真实协议数据集")

            # 分析perfection低的原因
            print(f"\n🔍 Perfection低的可能原因分析:")
            print(f"   1. 算法检测的边界与真实边界不完全匹配")
            print(f"   2. 真实数据的字段边界可能非常复杂")
            print(f"   3. 算法可能过度分割或分割不足")
            print(f"   4. 协议特异性规则需要进一步优化")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FieldHunter真实数据实验')

    parser.add_argument('--data-root', type=str, default='../../Msg2',
                        help='数据根目录路径 (默认: ../../Msg2)')

    parser.add_argument('--protocols', nargs='+',
                        help='要测试的协议列表 (如果未指定，将测试所有可用协议)')

    parser.add_argument('--max-samples', type=int, default=None,
                        help='每个协议的最大样本数 (如果未指定，将使用该协议的所有数据)')

    args = parser.parse_args()

    try:
        # 检查数据目录
        if not Path(args.data_root).exists():
            print(f"❌ 数据根目录不存在: {args.data_root}")
            print("请确保数据目录结构如下:")
            print("../../Msg2/")
            print("└── csv/")
            print("    ├── smb/")
            print("    ├── dns/")
            print("    └── ...")
            return

        experiment = RealDataFieldHunterExperiment(data_root=args.data_root)

        print(f"🌟 FieldHunter真实数据实验设置:")
        print(f"   数据根目录: {args.data_root}")
        print(f"   可用协议: {experiment.protocols}")
        print(f"   测试协议: {args.protocols or '全部'}")
        print(f"   最大样本: {args.max_samples or '使用全部数据'}")

        experiment.run_experiments(protocols=args.protocols, max_samples=args.max_samples)
        experiment.generate_report()

        print("\n✅ FieldHunter真实数据实验完成！")

    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()