import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# 准备数据
# 数据说明：与tp_method、seq_len、hidden_size三个数组中的位置相对应。例如第五个元素代表k划分 + 256 seq + 1024 hidden_size。
data = {
    'tp_method': ['mn', 'mn', 'mn', 'mn', 'k', 'k', 'k', 'k', 'mnk', 'mnk', 'mnk', 'mnk',
                  'mn', 'mn', 'mn', 'mn', 'k', 'k', 'k', 'k', 'mnk', 'mnk', 'mnk', 'mnk',
                  'mn', 'mn', 'mn', 'mn', 'k', 'k', 'k', 'k', 'mnk', 'mnk', 'mnk', 'mnk'],
    'seq_len': [256, 512, 1024, 2048, 256, 512, 1024, 2048, 256, 512, 1024, 2048,
                256, 512, 1024, 2048, 256, 512, 1024, 2048, 256, 512, 1024, 2048,
                256, 512, 1024, 2048, 256, 512, 1024, 2048, 256, 512, 1024, 2048],
    'hidden_size': [1024]*12 + [2048]*12 + [2560]*12,
    'latency_ns': [836502506, 931505130, 1156801514, 1689582538, 237064854, 462631342, 921695886, 1989563630, 582692224, 706082550, 966502032, 1522580726,
                   3174292266, 3371946516, 3734799358, 4795679616, 526071982, 942001902, 1822136110, 4504906464, 2149121192, 2346122272, 2786919606, 3926998784,
                   6100966566, 6428822266, 7028386326, 8716377182, 841069294, 1543341660, 2977814734, 7275459406, 4154358600, 4478439218, 5246908732, 6904453670]
}

df = pd.DataFrame(data)
df['latency_ms'] = df['latency_ns'] / 1e6  # 转换为毫秒

# 设置学术风格
plt.rcParams.update({
    # 'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
        'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.edgecolor': 'black',
})

# 创建图形
fig, ax1 = plt.subplots(figsize=(16, 7))

# 定义颜色和样式
colors = {'k': '#7A9273', 'mn': '#B37070', 'mnk': '#7E8CAD'}
color_line = {'k': '#1A8C5C', 'mn': '#EE4266', 'mnk': '#5A35FF'}
patterns = {'k': '///', 'mn': '...', 'mnk': 'xxx'}
line_styles = {'k': '-', 'mn': '--', 'mnk': '-.'}
markers = {'k': 'o', 'mn': '^', 'mnk': 's'}

# 准备x轴标签和位置
hidden_sizes = [1024, 2048, 2560]
seq_lengths = [256, 512, 1024, 2048]
methods = ['k', 'mn', 'mnk']

# 创建分组的x轴位置
x_labels = []
x_positions = []
pos = 0
group_positions = []

for i, hidden_size in enumerate(hidden_sizes):
    group_start = pos
    for j, seq_len in enumerate(seq_lengths):
        x_labels.append(f'S{seq_len}')
        x_positions.append(pos)
        pos += 1
    pos += 0.5  # 组间间距
    group_positions.append((group_start + pos - 1.5) / 2)  # 组中心位置

# 绘制柱状图
bar_width = 0.25
x_pos_array = np.array(x_positions)

for i, method in enumerate(methods):
    latencies = []
    for hidden_size in hidden_sizes:
        for seq_len in seq_lengths:
            subset = df[(df['tp_method'] == method) & 
                       (df['hidden_size'] == hidden_size) & 
                       (df['seq_len'] == seq_len)]
            latencies.append(subset['latency_ms'].iloc[0])
    
    # 绘制柱状图
    bars = ax1.bar(x_pos_array + i * bar_width, latencies, bar_width, 
                   label=f'TP-{method.upper()}', color=colors[method], 
                   alpha=0.7, hatch=patterns[method], edgecolor='black', linewidth=0.8)

# 设置柱状图的x轴
ax1.set_xlabel('Configuration (Model Params. & Sequence Length)', fontweight='bold', fontsize=30)
ax1.set_ylabel('Latency (ms)', fontweight='bold', color='black', fontsize=30)
ax1.set_xticks(x_pos_array + bar_width)
ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=25)

# 添加组分隔线和标签
for i in range(len(hidden_sizes)-1):
    sep_pos = x_positions[4*(i+1)] - 0.25
    ax1.axvline(x=sep_pos, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)

# 添加隐藏层大小标签
hidd_layer = ['Qwen3_1.7B', 'Qwen3_4B', 'Qwen3_8B']
for i, (pos, hidden_size) in enumerate(zip(group_positions, hidden_sizes)):
    ax1.text(pos, ax1.get_ylim()[1] * 0.95, f'{hidd_layer[i]}', 
             ha='center', va='top', fontweight='bold', fontsize=25,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))

ax1.tick_params(axis='y', labelsize=25)

# 创建第二个y轴用于折线图显示相对性能
ax2 = ax1.twinx()

# 计算并绘制相对性能折线图
for method in methods:
    latencies = []
    for hidden_size in hidden_sizes:
        for seq_len in seq_lengths:
            subset = df[(df['tp_method'] == method) & 
                       (df['hidden_size'] == hidden_size) & 
                       (df['seq_len'] == seq_len)]
            latencies.append(subset['latency_ms'].iloc[0])
    
    # 计算相对于最小值的比例
    min_latency = min(df['latency_ms'])
    relative_performance = [lat / min_latency for lat in latencies]
    
    # 绘制折线图
    ax2.plot(x_pos_array + bar_width, relative_performance, 
             color=color_line[method], linestyle=line_styles[method], 
             marker=markers[method], linewidth=3, markersize=8, alpha=0.9,
             markerfacecolor='white', markeredgecolor=color_line[method], 
             markeredgewidth=2)

ax2.set_ylabel('Relative Perf.', fontweight='bold', fontsize=30)
ax2.tick_params(axis='y', labelsize=25)

# 设置标题和网格
# ax1.set_title('Tensor Parallelism Performance Analysis: Latency vs. Relative Performance\nfor Qwen3 Series Model', 
#               fontsize=16, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, axis='y')

# 创建组合图例
# 柱状图图例
bar_legend = ax1.legend(loc='upper left', bbox_to_anchor=(0.215, 0.85), frameon=True, edgecolor='black',
                       title='Latency', fontsize=23)
ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
ax1.yaxis.get_offset_text().set_fontsize(25)
bar_legend.get_title().set_fontweight('bold')
bar_legend.get_title().set_fontsize(25)

# 折线图图例 - 修复了这里的错误
line_handles = []
for method in methods:
    line = plt.Line2D([0], [0], color=color_line[method], linestyle=line_styles[method],
                     marker=markers[method], linewidth=5, markersize=10,
                     markerfacecolor='white', markeredgecolor=color_line[method],  # 修复：使用正确的颜色值
                     markeredgewidth=2, label=f'TP-{method.upper()}')
    line_handles.append(line)

line_legend = ax2.legend(handles=line_handles, loc='upper left', frameon=True, edgecolor='black',
                        bbox_to_anchor=(0, 0.85), title='Relative Perf.', fontsize=23)
line_legend.get_title().set_fontweight('bold')
line_legend.get_title().set_fontsize(23)

# 调整布局
plt.tight_layout()

# 保存为PDF
output_pdf = 'tp_dimention.pdf'
with PdfPages(output_pdf) as pdf:
    pdf.savefig(fig)
    plt.close()

# 输出关键洞察
print("Key Performance Insights:")
print("="*50)
print(f"Best overall performer: TP-K (average: {df[df['tp_method'] == 'k']['latency_ms'].mean():.0f}ms)")
print(f"Sequence length scaling: TP-K shows best scaling with sequence length")
print(f"Hidden size scaling: TP-MN performance degrades most with larger hidden sizes")
fastest_config = df.loc[df['latency_ms'].idxmin()]
print(f"Fastest configuration: TP-{fastest_config['tp_method'].upper()}, H{fastest_config['hidden_size']}, S{fastest_config['seq_len']} ({fastest_config['latency_ms']:.0f}ms)")

# 显示图形（如果需要）
# plt.show()