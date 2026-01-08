import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def draw_figure2():
    # 1. [修复字体] 使用通用无衬线字体，避免 "Arial not found"
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
    
    # 2. [调整画布] 
    # 取消 constrained_layout，改用手动调整 hspace
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    
    # 3. [修复间距] hspace=0.3 让子图更紧凑 (之前可能是默认的 0.5+)
    plt.subplots_adjust(hspace=0.35, top=0.93, bottom=0.08, left=0.08, right=0.95)
    
    # --- 数据模拟 (保持不变) ---
    total_len = 60
    header_len = 10
    x = np.arange(total_len)
    
    # 顶层数据
    entropy = np.random.rand(total_len) * 0.8 + 0.2
    entropy[:header_len] = [0.1, 0.1, 0.9, 0.2, 0.8, 0.8, 0.8, 0.8, 0.1, 0.1]
    
    alignment = np.zeros(total_len)
    alignment[:header_len] = 0.95
    alignment[header_len:] = np.random.rand(total_len - header_len) * 0.2
    
    decay = np.ones(total_len)
    decay[header_len+2:] = np.exp(-0.3 * np.arange(total_len - (header_len+2)))

    # --- (a) 统计层 ---
    ax0 = axes[0]
    ax0.plot(x, entropy, color='gray', alpha=0.4, label='Entropy (Noisy)', linestyle='--')
    ax0.plot(x, alignment, color='#1f77b4', linewidth=2, label='Alignment Score')
    ax0.plot(x, decay, color='#d62728', linewidth=2.5, label='Safe-Zone Decay')
    
    # 标注 Safe Zone
    rect = patches.Rectangle((0, 0), header_len+2, 1.1, linewidth=0, edgecolor='none', facecolor='#2ca02c', alpha=0.1)
    ax0.add_patch(rect)
    ax0.text(5, 0.8, 'Safe Zone', ha='center', color='#2ca02c', fontweight='bold', fontsize=9)
    
    ax0.set_ylabel('Score')
    ax0.set_title('(a) Statistical Features & Spatial Decay', loc='left', fontweight='bold', fontsize=11)
    ax0.legend(loc='upper right', fontsize=8, frameon=False)
    ax0.set_ylim(0, 1.2)
    ax0.set_yticks([])

    # --- (b) 神经层 ---
    ax1 = axes[1]
    attention = np.zeros(total_len)
    attention[:header_len] = np.random.rand(header_len) * 0.5 + 0.3
    # CRC 位置
    attention[26:28] = 0.9
    attention[42:44] = 0.9
    
    im = ax1.imshow(attention.reshape(1, -1), aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax1.set_title('(b) BERT-MLM Semantic Attention', loc='left', fontweight='bold', fontsize=11)
    ax1.set_yticks([])
    
    # 标注 CRC
    ax1.annotate('CRC', xy=(27, 0.5), xytext=(27, -2), arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.5), ha='center', fontsize=9)
    
    # --- (c) 错误分割 ---
    ax2 = axes[2]
    # 模拟过分割
    cuts_without = [0] + [i for i in range(1, total_len) if entropy[i] > 0.6] + [total_len]
    ax2.barh(0, total_len, height=0.6, color='#f8d7da', edgecolor='black') # 浅红色背景
    for c in cuts_without:
        ax2.axvline(x=c, color='#721c24', linestyle='-', linewidth=1)
    
    ax2.text(total_len/2, 0, 'Without Safe-Zone: Over-segmented', ha='center', va='center', color='#721c24', fontweight='bold')
    ax2.set_title('(c) Baseline Segmentation (No Decay)', loc='left', fontweight='bold', fontsize=11)
    ax2.set_yticks([])
    ax2.set_xlim(0, total_len)
    
    # --- (d) 正确分割 ---
    ax3 = axes[3]
    cuts_with = [0, 2, 3, 4, 6, 8, 10, total_len]
    ax3.barh(0, total_len, height=0.6, color='#d4edda', edgecolor='black') # 浅绿色背景
    for c in cuts_with:
        ax3.axvline(x=c, color='#155724', linestyle='-', linewidth=2)
        
    ax3.text(total_len/2, 0, 'With Safe-Zone: Clean Segmentation', ha='center', va='center', color='#155724', fontweight='bold')
    ax3.set_title('(d) NeuPRE Segmentation (Ours)', loc='left', fontweight='bold', fontsize=11)
    ax3.set_yticks([])
    ax3.set_xlabel('Byte Offset', fontsize=10)
    
    # 标注 Header 和 Payload
    ax3.annotate('Header', xy=(5, -0.6), ha='center', fontweight='bold', fontsize=9)
    ax3.annotate('Payload (Protected)', xy=(35, -0.6), ha='center', color='gray', fontsize=9)

    # 4. [修复显示] 只保存不显示
    save_path = 'figure2_safe_zone.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 2 generated successfully as '{save_path}'")
    # plt.show() <--- 已注释，避免在服务器上报错

if __name__ == '__main__':
    draw_figure2()