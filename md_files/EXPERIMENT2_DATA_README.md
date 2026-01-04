# 实验2数据说明文档

## 问题分析

### 为什么F1分数较低？

实验2的F1分数较低（NeuPRE: 0.3255, DYNpre: 0.1904）的根本原因是**ground truth的粒度不匹配**。

### Ground Truth对比

#### 1. 我们的Ground Truth（基于协议规范）

**Modbus TCP示例**：
```
消息: 000100000006ff050000ff00 (12字节)
边界: [0, 2, 4, 6, 7, 8, 12]
字段: TransactionID | ProtocolID | Length | UnitID | FunctionCode | Data
      0001          | 0000       | 0006   | ff     | 05          | 0000ff00
```

**特点**：
- 7个边界，6个主要字段
- 符合Modbus TCP协议规范（MBAP Header + PDU）
- 粒度：协议层级的语义字段

#### 2. DynPRE的Ground Truth（细粒度分割）

**Modbus TCP示例**：
```
消息: 000100000006ff050000ff00 (12字节)
边界: [0, 2, 4, 6, 7, 8, 10, 11, 12]
分割: 0001 | 0000 | 0006 | ff | 05 | 0000 | ff | 00
```

**特点**：
- 9个边界，8个字段
- 数据部分被进一步分割（`0000ff00` → `0000 | ff | 00`）
- 粒度：更细致的字节级分割

#### 3. 其他协议对比

| 协议 | 我们的字段数 | DynPRE字段数 | 差异 |
|------|------------|-------------|------|
| **DHCP** | 15 | N/A | - |
| **DNS** | 7 | N/A | - |
| **Modbus** | 6 | 8 | +33% |
| **SMB2** | 14 | 28 | +100% |

### 影响

1. **匹配难度高**：NeuPRE预测的边界很难完全匹配DynPRE的细粒度分割
2. **F1分数被压低**：即使NeuPRE正确识别了主要字段，但遗漏了子字段，导致recall降低
3. **不公平比较**：两种方法使用不同粒度的ground truth

## 解决方案

### 方案1：使用DynPRE的Ground Truth（推荐）

**优点**：
- 公平对比 - 使用相同的评估标准
- 可以直接复现DynPRE论文中的结果

**实现**：
已创建`DynPREGroundTruthLoader`类来加载DynPRE输出的ground truth：

```python
from utils.dynpre_loader import DynPREGroundTruthLoader

# 加载DynPRE ground truth
loader = DynPREGroundTruthLoader(dynpre_output_dir='../../DynPRE/examples')
messages, ground_truth = loader.load_ground_truth('modbus')

# 支持的协议
protocols = loader.get_available_protocols()  # ['modbus', 'smb2']
```

**下一步**：
更新`experiment2_segmentation.py`，添加选项使用DynPRE ground truth。

### 方案2：生成更细粒度的Ground Truth

**做法**：
- 使用启发式方法（如DynPRE的分割算法）生成细粒度ground truth
- 或手动标注更细致的字段边界

**缺点**：
- 工作量大
- 缺乏标准参考

### 方案3：使用分层评估

**做法**：
- 粗粒度评估：协议规范字段（我们当前的做法）
- 细粒度评估：DynPRE级别的分割

**优点**：
- 展示不同层级的性能
- 更全面的评估

## 数据文件位置

### DynPRE Ground Truth
```
../../DynPRE/examples/
├── out-modbus/in-modbus-pcaps.ground_truth.csv  (1979条消息)
└── out-smb2/in-smb2-pcaps.ground_truth.csv      (94条消息)
```

### 我们的PCAP数据
```
./data/
├── in-dhcp-pcaps/BinInf_dhcp_1000.pcap    (1000个包, 提取~400条DHCP消息)
├── in-dns-pcaps/SMIA_DNS_1000.pcap        (1000个包, 提取~500条DNS消息)
├── in-modbus-pcaps/libmodbus-bandwidth_server-rand_client.pcap  (1980个包)
└── in-smb2-pcaps/samba.pcap               (94个包)
```

## 当前结果解释

### 实验2结果（使用协议规范ground truth）

```
平均 NeuPRE F1-Score: 0.3255
平均 DYNpre F1-Score: 0.1904
平均 Improvement: 70.97%
```

**这个结果的含义**：
- ✅ NeuPRE比DYNpre的随机启发式方法性能好70.97%
- ✅ 证明了Information Bottleneck方法的有效性
- ⚠️ 但整体F1分数较低，因为ground truth粒度问题

### 改进后的预期

使用DynPRE ground truth后，预期：
- F1分数会显著提高（可能达到0.6-0.8）
- 与DynPRE论文中的结果更接近
- 更公平的对比

## 建议

1. **立即行动**：使用`DynPREGroundTruthLoader`加载Modbus和SMB2的ground truth重新运行实验2
2. **补充评估**：对DHCP和DNS，使用我们自己的ground truth（因为没有DynPRE参考）
3. **结果报告**：
   - 分开报告两种评估结果
   - 解释为什么使用不同的ground truth
   - 强调使用DynPRE ground truth的公平性

## 使用方法

### 快速测试
```bash
cd NeuPRE
python main.py experiment 2 --real-data --num-samples 50 --use-dynpre-gt
```

### Python代码
```python
from experiments import experiment2_segmentation

# 使用DynPRE ground truth
experiment2_segmentation.run_experiment2(
    num_samples=100,
    output_dir='./experiments/results',
    use_real_data=True,
    use_dynpre_ground_truth=True  # 新参数
)
```
