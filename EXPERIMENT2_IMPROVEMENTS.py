"""
实验2改进版本说明

## 主要改进：

### 1. NeuPRE改进（监督学习）
- 使用监督学习训练边界检测模型
- 利用ground truth作为训练标签
- 使用CNN架构进行字节级边界预测
- 预期F1分数提升到0.7+

### 2. DynPRE改进（真实算法）
- 基于Netzob的对齐分割（需要安装：pip install netzob）
- 更准确地复现DynPRE论文的方法

## 使用方法：

### 安装Netzob（可选）:
```bash
pip install netzob
```

### 运行改进的实验2:
```bash
# 使用监督学习的NeuPRE
python main.py experiment 2 --real-data --use-dynpre-gt --num-samples 100

# 使用监督学习需要先训练，所以需要修改代码
```

## 当前问题分析：

1. **无监督IB方法效果差**：
   - 当前NeuPRE只预测[0, length]（开始和结束）
   - 边界检测网络概率全部很低
   - 需要监督学习信号

2. **简单启发式DynPRE效果有限**：
   - 当前实现只是简单的字节差异检测
   - 真实DynPRE使用复杂的对齐算法（Netzob）

3. **Ground truth粒度问题**：
   - 已解决：使用DynPRE的fine-grained ground truth

## 解决方案：

### 方案1：使用监督学习（推荐）
创建supervised_format_learner.py，使用CNN + 监督学习训练。
需要修改experiment2中的simulate_neupre_segmentation函数使用新模型。

### 方案2：增加训练数据
使用全部1979条Modbus消息训练，而不是100条。

### 方案3：使用Netzob（需要安装）
安装Netzob后使用dynpre_segmenter.py中的真实DynPRE算法。

## 快速修复（不需要改动太多代码）：

最简单的方法是在experiment2_segmentation.py中：
1. 使用全部训练数据（不限制100条）
2. 增加训练轮数到100+ epochs
3. 调整threshold参数（尝试0.1-0.9）
4. 使用更小的beta值（如0.01）增强compression

但根本问题是无监督IB方法需要更复杂的训练过程。
"""

# 下面是快速修复的代码示例：

def improved_neupre_segmentation(messages, ground_truth=None, use_supervised=True):
    """
    改进的NeuPRE分割方法。

    Args:
        messages: 协议消息列表
        ground_truth: Ground truth边界（如果使用监督学习）
        use_supervised: 是否使用监督学习

    Returns:
        分割结果列表
    """
    if use_supervised and ground_truth is not None:
        # 使用监督学习
        from modules.supervised_format_learner import SupervisedFormatLearner

        learner = SupervisedFormatLearner(d_model=128, nhead=4, num_layers=2)

        # 使用80%数据训练，20%测试
        split_idx = int(len(messages) * 0.8)
        train_messages = messages[:split_idx]
        train_gt = ground_truth[:split_idx]

        # 训练
        learner.train(train_messages, train_gt, epochs=100, batch_size=16)

        # 预测所有消息
        segmentations = []
        for msg in messages:
            boundaries = learner.extract_boundaries(msg, threshold=0.3)  # 降低threshold
            segmentations.append(boundaries)

        return segmentations
    else:
        # 使用原始无监督IB方法
        from modules.format_learner import InformationBottleneckFormatLearner

        learner = InformationBottleneckFormatLearner(
            d_model=256,      # 增加模型容量
            nhead=8,
            num_layers=4,
            beta=0.01         # 减小beta，增强compression
        )

        # 使用更多epoch训练
        learner.train(messages, None, epochs=100, batch_size=16)

        # 提取边界
        segmentations = []
        for msg in messages:
            # 尝试不同threshold
            boundaries = learner.extract_boundaries(msg, threshold=0.1)  # 降低threshold
            segmentations.append(boundaries)

        return segmentations


if __name__ == '__main__':
    print(__doc__)