import matplotlib.pyplot as plt
import numpy as np

# 设置图表风格
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams.update({
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.edgecolor': 'black',
})

# 数据整理 - 32B模型
data_32B = {
    '128MB': {
        '32': [61881639662472, 57935772820112, 57937073693184],
        '64': [42504669237880, 36066390958752, 34139393636528],
        '128': [31327700440632, 25433070549384, 22668088473072],
    },
    '64MB': {
        # '64': [40514959975456, 34940264681800, 33805279245136],
        '128': [31649387951552, 25612070097640, 22820743433056],
    },
    '32MB': {
        # '64': [40809331521864, 35253278050480, 33949112717672],
        '128': [32996045718488, 26396259950448, 23212820256296],
    }
}

data_14B = {
    '128MB': {
        '32': [28996236525000, 27463783040000, 27463783040000],
        '64': [20021623360200, 17625399504496, 16869927038864],
        '128': [15605125875720, 13158069314048, 11930585197200],
    },
    '64MB': {
        '128': [15823704255816, 13251052648448, 11975463517496]
    },
    '32MB': {
        '128': [16072182871688, 13318637197016, 12020180557912]
    }
}

data_8B = {
    '128MB': {
        '32': [16886969918000, 16277711866000, 16277711866000],
        '64': [8592801380928, 7511942429176, 6999116937328],
        '128': [7228611937088, 6142044731128, 5603376660704],
    },
    '64MB': {
        # '64': [11921078354000, 10324362901000, 9961014130000],
        '128': [9336800164232, 7739143066952, 7072339677232]
    },
    '32MB': {
        # '64': [12120093014000, 10503242207000, 9963743586000],
        '128': [9536316542736, 7912599092304, 7071545976640]
    }
}

# 数据整理 - 4B模型
data_4B = {
    '128MB': {
        '32': [6403027583230, 6368525130000, 6368525130000],
        '64': [3930311107408, 3919711584776, 3919674079056],
        '128': [2726377943152, 2711544079464, 2706356772056],
    },
    '64MB': {
        '128': [3062884823640, 2823679493368, 2715347836768]
    },
    '32MB': {
        '128': [3699090238104, 3103597501432, 2811593563304]
    }
}

data_32B_moe = {
    '128MB': {
        '32': [7253094978088, 7214011971339, 7214011971339],
        '64': [4646238434152, 4633708151411, 4633663813822],
        '128': [3062983460171, 3046318170151, 3040490424650]
    }, 
    '64MB': {
        '128': [3524137525982, 3156719064316, 3035610203887]
    }, 
    '32MB': {
        '128': [4180738741413, 3423806867803, 3101675828476]
    }
}

# 组合所有模型数据
all_models = {
    '32B': data_32B,
    '14B': data_14B,
    '8B': data_8B,
    '4B': data_4B,
    '30BA3B': data_32B_moe,
}

# DRAM bandwidth values
dram_bw = [16, 32, 64]

# 创建图表
fig, ax = plt.subplots(1, 1, figsize=(24,6))

# 配色方案
model_colors = {
    '30BA3B': {'base': '#FF7F0E', 'dram': ['#CC5500', '#FF7F0E', '#FFA040']},
    '32B': {'base': '#7E8CAD', 'dram': ['#5F6F8A', '#7E8CAD', '#A0B0D8']}, 
    '14B': {'base': '#17becf', 'dram': ['#0e7a85', '#17becf', '#5cd3df']},   
    '8B':  {'base': '#A28CC2', 'dram': ['#846DA0', '#A28CC2', '#B8A8D4']},
    '4B': {'base': '#7A9273', 'dram': ['#5D7254', '#7A9273', '#9AA894']}   
}

# 柱状图参数
bar_width = 0.55
group_spacing = 1.2
model_spacing = 0.5

positions = []
all_values = []
all_colors = []
speedup_data = {}
x_labels = []
x_label_positions = []

x_pos = 0

# 用于存储每个组的中心位置和标签
all_group_centers = []
all_group_labels = []

# 遍历每个模型
for model_idx, (model_name, model_data) in enumerate(all_models.items()):
    model_start = x_pos
    
    # 按照顺序遍历：128MB, 64MB, 32MB
    for sram_name, sram_data in [('128MB', model_data['128MB']), 
                                  ('64MB', model_data['64MB']), 
                                  ('32MB', model_data['32MB'])]:
        for compute, values in sram_data.items():
            group_start = x_pos
            
            # 绘制每个DRAM带宽的柱子
            for i, (bw, value) in enumerate(zip(dram_bw, values)):
                positions.append(x_pos)
                all_values.append(value / 1e12)  # 转换为秒
                all_colors.append(model_colors[model_name]['dram'][i])
                x_pos += bar_width
            
            # 记录组中心位置
            group_center = group_start + bar_width * 1.0
            
            # 计算加速比
            baseline = values[0]
            speedup = [baseline / v for v in values]
            speedup_data[f'{model_name}_{sram_name}_{compute}'] = (group_center, speedup)
            
            # 为每个模型都添加x轴标签
            # 格式化SRAM大小（去掉MB）
            sram_size = sram_name.replace('MB', '')
            label = f'S{sram_size}A{compute}'
            all_group_centers.append(group_center)
            all_group_labels.append(label)
            
            x_pos += group_spacing
        
    # 模型之间的间隔
    x_pos += model_spacing

# 绘制柱状图
bars = ax.bar(positions, all_values, width=bar_width*0.9, color=all_colors, 
               edgecolor='black', linewidth=0.5, alpha=0.85)

# 创建第二个y轴用于加速比
ax2 = ax.twinx()

# 绘制加速比折线
for key, (center_x, speedup_values) in speedup_data.items():
    x_points = [center_x - bar_width, center_x, center_x + bar_width]
    
    # 解析key获取模型和SRAM信息
    parts = key.split('_')
    model = parts[0]
    sram = parts[1]
    
    # 根据模型选择颜色，根据SRAM选择线型
    linecolor = '#e74c3c'
    marker = 'o'
    
    if '128MB' in sram:
        linestyle = '-'
        linewidth = 2.5
    elif '64MB' in sram:
        linestyle = '--'
        linewidth = 2.2
    else:  # 32MB
        linestyle = ':'
        linewidth = 2.0
    
    ax2.plot(x_points, speedup_values, 
            color=linecolor, linestyle=linestyle, marker=marker,
            linewidth=linewidth, markersize=7, markeredgecolor='white',
            markeredgewidth=1.2, alpha=0.9, zorder=10)

# 设置主y轴（柱状图）
ax.set_ylabel('Latency (s)', fontsize=30, fontweight='bold')
ax.set_ylim(0, max(all_values) * 1.15)

# 设置第二y轴（折线图）
ax2.set_ylabel('Speedup', fontsize=30, fontweight='bold')
ax2.set_ylim(0.95, 1.7)
ax2.tick_params(axis='y', labelsize=25)

# 设置x轴 - 使用所有收集的标签
ax.set_xticks(all_group_centers)
ax.set_xticklabels(all_group_labels, fontsize=22, rotation=45, ha='right')
ax.set_xlabel('SRAM Size and Compute Configuration', fontsize=30, fontweight='bold')

ax.tick_params(axis='y', labelsize=25)
ax.tick_params(axis='x', labelsize=20)

# 添加网格
ax.grid(True, axis='y', linestyle='--', alpha=0.3, linewidth=0.5)

# 创建图例
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# DRAM带宽图例（使用渐变效果）
dram_legend_elements = []
# 使用32B模型的颜色作为示例
for i, bw in enumerate(dram_bw):
    rect = Rectangle((0,0), 1, 1, fc=model_colors['32B']['dram'][i], 
                    edgecolor='black', linewidth=0.5, alpha=0.85)
    dram_legend_elements.append(rect)

# 将DRAM带宽图例放在顶部左侧
legend2 = ax.legend(dram_legend_elements, 
                   [f'{bw} GB/s' for bw in dram_bw],
                   loc='lower left', title='DRAM bandwidth',
                   fontsize=24, title_fontsize=25,
                   frameon=True,
                   edgecolor='black',
                   bbox_to_anchor=(-0.01, 1.02),
                   ncol=3)

# 加速比图例
speedup_legend_elements = [
    Line2D([0], [0], color='#e74c3c', linewidth=2.5, linestyle='-', marker='o', markersize=7),
    Line2D([0], [0], color='#e74c3c', linewidth=2.2, linestyle='--', marker='o', markersize=7),
    Line2D([0], [0], color='#e74c3c', linewidth=2.0, linestyle=':', marker='o', markersize=7),
]

# 将加速比图例放在顶部右侧
legend3 = ax2.legend(speedup_legend_elements, 
                    ['128MB SRAM', '64MB SRAM', '32MB SRAM'],
                    loc='lower right', title='Speedup vs. original DRAM bandwidth',
                    fontsize=24, title_fontsize=25,
                    frameon=True,
                    edgecolor='black',
                    ncol=3,
                    bbox_to_anchor=(1.01, 1.02))

# 添加模型分隔线
model_boundaries = []
current_x = 0
for model_idx, (model_name, model_data) in enumerate(all_models.items()):
    if model_idx > 0:
        ax.axvline(x=current_x - model_spacing/2 - group_spacing/2, 
                  color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # 计算下一个模型的起始位置
    num_groups = sum(len(sram_data) for sram_data in model_data.values())
    current_x += num_groups * (3 * bar_width + group_spacing) + model_spacing

# 添加模型标签 - 调整位置避免与顶部图例重叠
model_label_y = max(all_values) * 0.9  # 降低模型标签位置
current_x = 0
for model_name, model_data in all_models.items():
    num_groups = sum(len(sram_data) for sram_data in model_data.values())
    model_center = current_x + num_groups * (3 * bar_width + group_spacing) / 2
    ax.text(model_center, model_label_y, f'Qwen3_{model_name}', 
           ha='center', va='bottom', fontsize=22, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    edgecolor=model_colors[model_name]['base'], linewidth=1.5))
    current_x += num_groups * (3 * bar_width + group_spacing) + model_spacing

# 添加图例到图表
ax.add_artist(legend2)

# 调整布局，为旋转的x轴标签留出空间
plt.subplots_adjust(left=0.08, bottom=0.3, top=0.77)  # 调整top值为图例留出空间

fig.savefig('qwen_bundle_20.pdf', format='pdf')

# 显示图表
# plt.show()