import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置学术风格
plt.rcParams.update({
    'font.size': 11,
    # 'font.family': 'serif',
    'axes.linewidth': 1.2,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.edgecolor': 'black',
})

# 原始数据重新整理
data = {
    # TP4配置
    ('4B', 'TP4'): {'linear-seq': 3096781356336, 'linear-interleave': 3103597501432, 'mesh': 2654077009608, 'ring': 2653918999736},
    ('8B', 'TP4'): {'linear-seq': 7723939355336, 'linear-interleave': 7739143066952, 'mesh': 6626662710744, 'ring': 6626515258840},
    ('14B', 'TP4'): {'linear-seq': 13237413817984, 'linear-interleave': 13251052648448, 'mesh': 11227448009744, 'ring': 11227428826128},
    ('32B', 'TP4'): {'linear-seq': 28342358157432, 'linear-interleave': 29044145083128, 'mesh': 24544866636952, 'ring': 24544617923128},
    # TP16配置
    ('4B', 'TP16'): {'linear-seq': 2016260619016*2.25, 'linear-interleave': 2292628503368*2.25, 'mesh': 1875789295416*2.25, 'ring': 1783471877272*2.25},
    ('8B', 'TP16'): {'linear-seq': 5117723302416*2.25, 'linear-interleave': 5874787866656*2.25, 'mesh': 4801151673272*2.25, 'ring': 4563761690560*2.25},
    ('14B', 'TP16'): {'linear-seq': 7198697055744*2.5, 'linear-interleave': 8321854554300*2.5, 'mesh': 6739225041480*2.5, 'ring': 6374497972560*2.5},
    ('32B', 'TP16'): {'linear-seq': 9786089293540*4, 'linear-interleave': 11547585362720*4, 'mesh': 9259841090688*4, 'ring': 8753395376208*4}
}

# 转换数据
rows = []
for (model_size, tp_config), strategies in data.items():
    for strategy, latency_ps in strategies.items():
        rows.append({
            'model_size': model_size,
            'tp_config': tp_config,
            'strategy': strategy,
            'latency_ps': latency_ps,
            'latency_ms': latency_ps / 1e9
        })

df = pd.DataFrame(rows)

# 创建图形
fig, ax1 = plt.subplots(figsize=(16, 5))

# 定义颜色和样式
colors = {
    'linear-interleave': '#B37070',   # 红色
    'linear-seq': '#7E8CAD',
    'mesh': '#7A9273',     # 橙色
    'ring': '#A28CC2'      # 绿色
}

# 模型配置
model_sizes = ['4B', '8B', '14B', '32B']
model_names = ['Qwen3_4B', 'Qwen3_8B', 'Qwen3_14B', 'Qwen3_32B']
strategies = ['linear-interleave', 'linear-seq', 'mesh', 'ring']
tp_configs = ['TP4', 'TP16']

# 设置x轴位置 - 每个模型有6个柱子（3个策略 × 2个TP配置）
n_models = len(model_sizes)
n_bars_per_model = len(strategies) * len(tp_configs)
bar_width = 0.08
model_spacing = 1.0  # 模型间距

# 计算每个模型组的中心位置
model_centers = np.arange(n_models) * model_spacing
all_positions = []
all_labels = []
bar_positions = {}

# 为每个模型创建柱状图
for model_idx, model_size in enumerate(model_sizes):
    model_center = model_centers[model_idx]
    
    # 在每个模型内部，按TP4, TP16分组
    for tp_idx, tp_config in enumerate(tp_configs):
        for strategy_idx, strategy in enumerate(strategies):
            # 计算具体位置
            bar_position = (model_center - 
                          (n_bars_per_model * bar_width) / 2 + 
                          (tp_idx * len(strategies) + strategy_idx) * bar_width + 
                          bar_width / 2)
            
            # 存储位置信息
            key = (model_size, tp_config, strategy)
            bar_positions[key] = bar_position
            
            # 获取延迟数据
            subset = df[(df['model_size'] == model_size) & 
                       (df['tp_config'] == tp_config) & 
                       (df['strategy'] == strategy)]
            latency = subset['latency_ms'].values[0]
            
            # 设置柱子样式
            hatch = None if tp_config == 'TP4' else '///'
            alpha = 0.9 if tp_config == 'TP4' else 0.7
            
            # 绘制柱状图
            bar = ax1.bar(bar_position, latency, bar_width,
                         color=colors[strategy], alpha=alpha,
                         hatch=hatch, edgecolor='black', linewidth=0.5)
            
            all_positions.append(bar_position)
            all_labels.append(f'{model_size}-{tp_config}-{strategy}')

# 设置左侧y轴
ax1.set_xlabel('Model Size', fontweight='bold', fontsize=30)
ax1.set_ylabel('Latency (ms)', fontweight='bold', fontsize=30, color='black')
ax1.tick_params(axis='y', labelcolor='black',labelsize=25)
ax1.grid(True, alpha=0.3, axis='y')
ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
ax1.yaxis.get_offset_text().set_fontsize(25)
ax1.tick_params(axis='x', labelcolor='black',labelsize=25)
# 设置x轴标签
ax1.set_xticks(model_centers)
ax1.set_xticklabels(model_sizes, fontweight='bold')

# 添加TP4/TP16分组标签
# for model_idx, model_size in enumerate(model_sizes):
#     model_center = model_centers[model_idx]
    
#     # TP4标签
#     tp4_center = model_center - bar_width * 1.5
#     ax1.text(tp4_center, ax1.get_ylim()[0] - (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.05,
#              'TP4', ha='center', va='top', fontsize=9, fontweight='bold')
    
#     # TP16标签
#     tp16_center = model_center + bar_width * 1.5
#     ax1.text(tp16_center, ax1.get_ylim()[0] - (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.05,
#              'TP16', ha='center', va='top', fontsize=9, fontweight='bold')

# 创建第二个y轴用于加速比
ax2 = ax1.twinx()

# 绘制加速比折线 - 每个模型的每个TP配置都有独立的折线
line_markers = {'TP4': 'o', 'TP16': 's'}
line_styles = {'TP4': '-', 'TP16': '--'}

for model_idx, model_size in enumerate(model_sizes):
    model_center = model_centers[model_idx]
    
    for tp_idx, tp_config in enumerate(tp_configs):
        # 获取该模型该TP配置下的所有策略数据
        linear_latency = df[(df['model_size'] == model_size) & 
                           (df['tp_config'] == tp_config) & 
                           (df['strategy'] == 'linear-interleave')]['latency_ms'].values[0]
        
        linear_seq_latency = df[(df['model_size'] == model_size) & 
                           (df['tp_config'] == tp_config) & 
                           (df['strategy'] == 'linear-seq')]['latency_ms'].values[0]
        
        mesh_latency = df[(df['model_size'] == model_size) & 
                         (df['tp_config'] == tp_config) & 
                         (df['strategy'] == 'mesh')]['latency_ms'].values[0]
        
        ring_latency = df[(df['model_size'] == model_size) & 
                         (df['tp_config'] == tp_config) & 
                         (df['strategy'] == 'ring')]['latency_ms'].values[0]
        
        # 计算加速比
        speedups = [
            1.0, 
            linear_latency / linear_seq_latency, 
            linear_latency / mesh_latency, 
            linear_latency / ring_latency
        ]
        
        # 使用柱状图的实际x位置
        x_positions = [
            bar_positions[(model_size, tp_config, 'linear-interleave')],
            bar_positions[(model_size, tp_config, 'linear-seq')],
            bar_positions[(model_size, tp_config, 'mesh')],
            bar_positions[(model_size, tp_config, 'ring')]
        ]
        
        # 绘制折线
        line_color = 'blue' if tp_config == 'TP4' else 'purple'
        ax2.plot(x_positions, speedups, 
                color=line_color, linestyle=line_styles[tp_config],
                linewidth=2.5, marker=line_markers[tp_config], markersize=6,
                label=f'Speedup {tp_config}' if model_idx == 0 else "")
        
        # 在显著的加速比点上添加标注
        # for i, (x_pos, speedup) in enumerate(zip(x_positions, speedups)):
        #     if speedup > 1.1:  # 只标注显著的加速比
        #         ax2.annotate(f'{speedup:.2f}×', 
        #                    xy=(x_pos, speedup), 
        #                    xytext=(x_pos, speedup * 1.1),
        #                    ha='center', va='bottom',
        #                    fontsize=16, fontweight='bold',
        #                    bbox=dict(boxstyle='round,pad=0.2', 
        #                            facecolor='white', alpha=0.8, edgecolor='gray'))

# 设置右侧y轴
ax2.set_ylabel('Speedup vs.\nInterleave', fontweight='bold', fontsize=30, color='black')
ax2.tick_params(axis='y',labelsize=25)
ax2.set_ylim(0.8, 3.2)
ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)

# 创建图例
# 延迟条形图图例
latency_handles = [plt.Rectangle((0,0),1,1, color=colors[strategy], alpha=0.9, label=f'{strategy.capitalize()}')
                  for strategy in strategies]
style_handles = [plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.9, label='TP4'),
                plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.7, hatch='///', label='TP16')]

# 加速比折线图例
speedup_handles = ax2.get_legend_handles_labels()[0]

# 分别放置图例
legend1 = ax1.legend(latency_handles, [h.get_label() for h in latency_handles],
                    loc='upper left', title='Core Placement', 
                    frameon=True, edgecolor='black',fontsize=22)

legend2 = ax1.legend(style_handles, [h.get_label() for h in style_handles],
                    loc='upper left', bbox_to_anchor=(0.59, 1), title='TP',
                    frameon=True, edgecolor='black',fontsize=25)

legend3 = ax2.legend(speedup_handles, [f'{tp}' for tp in tp_configs],
                    loc='upper left', bbox_to_anchor=(0.315, 1), title='Speedup Ratio',
                    frameon=True, edgecolor='black',fontsize=25)

# 添加第一个图例回来
ax1.add_artist(legend1)
legend1.get_title().set_fontweight('bold')
legend1.get_title().set_fontsize(25)
legend2.get_title().set_fontweight('bold')
legend2.get_title().set_fontsize(25)
legend3.get_title().set_fontweight('bold')
legend3.get_title().set_fontsize(25)

# 标题和总结
# plt.title('Communication Strategy Performance: Latency and Speedup Analysis by Model Size and TP Configuration\n' +
#          'Ring Consistently Outperforms Mesh and Linear Across All Configurations', 
#          fontsize=14, fontweight='bold', pad=25)

# 添加性能洞察
# insight_text = ('Performance Summary:\n'
#                '• Ring strategy shows consistent superiority (up to 2.67× speedup)\n'
#                '• TP16 demonstrates larger performance gaps than TP4\n'
#                '• Larger models benefit more from optimized communication strategies\n'
#                '• Mesh provides moderate but consistent improvement over Linear')

# plt.figtext(0.02, 0.02, insight_text, fontsize=10,
#            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9))

plt.subplots_adjust(left=0.06, bottom=0.2, top=0.9) 
fig.savefig('tp_mapping.pdf', format='pdf')
# 详细性能报告
print("Comprehensive Performance Analysis:")
print("=" * 70)

for model_size in model_sizes:
    print(f"\n{model_size} Model Analysis:")
    print("-" * 40)
    
    for tp_config in tp_configs:
        print(f"\n  {tp_config} Configuration:")
        
        linear_latency = df[(df['model_size'] == model_size) & 
                           (df['tp_config'] == tp_config) & 
                           (df['strategy'] == 'linear-interleave')]['latency_ms'].values[0]
        
        mesh_latency = df[(df['model_size'] == model_size) & 
                         (df['tp_config'] == tp_config) & 
                         (df['strategy'] == 'mesh')]['latency_ms'].values[0]
        
        ring_latency = df[(df['model_size'] == model_size) & 
                         (df['tp_config'] == tp_config) & 
                         (df['strategy'] == 'ring')]['latency_ms'].values[0]
        
        mesh_speedup = linear_latency / mesh_latency
        ring_speedup = linear_latency / ring_latency
        
        print(f"    Linear: {linear_latency:>10.1f} ms (baseline)")
        print(f"    Mesh:   {mesh_latency:>10.1f} ms ({mesh_speedup:>5.2f}× speedup)")
        print(f"    Ring:   {ring_latency:>10.1f} ms ({ring_speedup:>5.2f}× speedup)")
        
        improvement = ((linear_latency - ring_latency) / linear_latency) * 100
        print(f"    → Ring improves over Linear by {improvement:.1f}%")