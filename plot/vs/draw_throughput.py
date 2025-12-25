import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
import seaborn as sns

# 设置学术会议风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'axes.grid': True,
    'grid.linestyle': '--'
})

# 设置颜色调色板 - 冷色调柱状图，暖色调折线
# 冷色调 (蓝、青、紫、绿系)
bar_colors = ['#1f77b4', '#2ca02c', '#17becf', '#9467bd', '#3f8fbf', '#0f5f7f', '#1a8c5c', '#8c564b', '#2f5f8f', '#5f9ea0']
# 暖色调 (红、橙、黄、粉系) 
line_colors = ["#5a35ff", '#f7931e', '#ffd23f', '#ee4266', '#f25c78', '#ff8c42', '#ffb347', '#ff7b7b', '#ff9f40', '#ffa726']

def parse_filename(filepath):
    """从文件路径中提取场景名"""
    filename = os.path.basename(filepath)
    name_without_ext = filename.replace('.json', '')
    parts = name_without_ext.split('_')
    
    # 场景名是最后两个部分 (例如: 1_1, 2_2等)
    if len(parts) >= 2:
        scenario_name = '_'.join(parts[-2:])
    else:
        scenario_name = 'unknown'
    
    return scenario_name

def load_and_process_data(json_files):
    """加载和处理JSON数据"""
    AREA = [749.4, 749.4, 761.7, 648.45]  # 常量
    
    # 按文件和场景组织数据: {file_key: {scenario: data}}
    file_data = {}
    all_scenarios = set()
    cnt = 0
    
    for file_key, data_dict in json_files.items():
        file_data[file_key] = {}
        
        for file_path, metrics in data_dict.items():
            scenario_name = parse_filename(file_path)
            
            # 提取需要的数据
            tbt_mean = metrics['tbt']['mean'] / 1e9  # 转换为秒
            throughput = metrics['throughput_tokens_per_second']
            throughput_per_area = throughput / AREA[cnt]
            
            file_data[file_key][scenario_name] = {
                'tbt_mean': tbt_mean,
                'throughput': throughput,
                'throughput_per_area': throughput_per_area,
                'file_path': file_path
            }
            
            all_scenarios.add(scenario_name)
        
        cnt += 1
    
    return file_data, sorted(all_scenarios, key=lambda x: (-int(x.split('_')[1]), int(x.split('_')[0])))

def create_grouped_plot(file_data, scenarios):
    """创建按场景分组的组合图表"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # 创建双y轴
    ax2 = ax.twinx()
    
    # 计算布局参数
    n_files = len(file_data)
    n_scenarios = len(scenarios)
    
    # 每个场景组的宽度和柱子宽度
    group_width = 0.8
    bar_width = group_width / n_files
    group_spacing = 0.3  # 场景组之间的间隔
    
    # 计算每个场景组的中心位置
    group_positions = np.arange(n_scenarios) * (1 + group_spacing)
    
    # 为每个文件分配颜色
    file_keys = list(file_data.keys())
    file_color_map = {file_key: bar_colors[i % len(bar_colors)] 
                      for i, file_key in enumerate(file_keys)}
    line_color_map = {file_key: line_colors[i % len(line_colors)] 
                      for i, file_key in enumerate(file_keys)}
    
    # 绘制柱状图 (显示 throughput)
    for file_idx, file_key in enumerate(file_keys):
        throughput_values = []
        bar_positions = []
        
        for scenario_idx, scenario in enumerate(scenarios):
            # 计算该文件在该场景组中的位置
            group_center = group_positions[scenario_idx]
            bar_pos = group_center + (file_idx - (n_files-1)/2) * bar_width
            
            if scenario in file_data[file_key]:
                throughput_value = file_data[file_key][scenario]['throughput']
                throughput_values.append(throughput_value)
                bar_positions.append(bar_pos)
        
        # 绘制该文件的所有柱子
        if throughput_values:  # 只有当有数据时才绘制
            bars = ax.bar(bar_positions, throughput_values, bar_width, 
                         label=f'File {file_key} (Throughput)', 
                         color=file_color_map[file_key], alpha=0.8, 
                         edgecolor='black', linewidth=0.8)
            
            # 添加数值标签
            for bar, value in zip(bars, throughput_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max([max(file_data[fk].get(s, {'throughput': 0})['throughput'] for s in scenarios) for fk in file_keys])*0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    # 绘制折线图（每个文件一条线）
    for file_idx, file_key in enumerate(file_keys):
        line_x = []
        line_y = []
        
        for scenario_idx, scenario in enumerate(scenarios):
            if scenario in file_data[file_key]:
                # 使用场景组的中心位置作为折线的x坐标
                line_x.append(group_positions[scenario_idx])
                line_y.append(file_data[file_key][scenario]['throughput_per_area'])
        
        # 绘制折线
        if len(line_x) > 0:  # 只有当有数据点时才绘制线
            ax2.plot(line_x, line_y, marker='o', linewidth=3, markersize=10,
                    label=f'File {file_key} (Throughput/Area)', 
                    color=line_color_map[file_key],
                    markerfacecolor='white', markeredgecolor=line_color_map[file_key],
                    markeredgewidth=2, zorder=10)
    
    # 设置x轴
    ax.set_xticks(group_positions)
    ax.set_xticklabels([f'Scenario {s}' for s in scenarios], fontweight='bold')
    
    # 在场景组之间添加分隔线
    for i in range(len(group_positions) - 1):
        separator_x = (group_positions[i] + group_positions[i+1]) / 2
        ax.axvline(x=separator_x, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # 设置标签和标题
    ax.set_xlabel('Scenarios', fontweight='bold', fontsize=16)
    ax.set_ylabel('Throughput (tokens/s)', fontweight='bold', fontsize=14, color='steelblue')
    ax2.set_ylabel('Throughput/Area (tokens/s/unit)', fontweight='bold', fontsize=14, color='darkred')
    
    ax.set_title('Performance Analysis: Throughput and Throughput/Area by Scenario', 
                fontweight='bold', pad=20, fontsize=18)
    
    # 设置y轴颜色和范围
    ax.tick_params(axis='y', labelcolor='steelblue', labelsize=12)
    ax2.tick_params(axis='y', labelcolor='darkred', labelsize=12)
    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 创建图例
    # Throughput柱状图图例
    bar_legend_elements = []
    for file_key in file_keys:
        bar_legend_elements.append(plt.Rectangle((0,0),1,1, 
                                                fc=file_color_map[file_key], 
                                                alpha=0.8, 
                                                label=f'File {file_key} (Throughput)'))
    
    # 折线图图例
    line_legend_elements = []
    for file_key in file_keys:
        line_legend_elements.append(plt.Line2D([0], [0], 
                                             color=line_color_map[file_key], 
                                             linewidth=3, marker='o', markersize=8,
                                             markerfacecolor='white',
                                             markeredgecolor=line_color_map[file_key],
                                             label=f'File {file_key} (T/A)'))
    
    # 放置图例
    legend1 = ax.legend(handles=bar_legend_elements, loc='upper left', 
                       frameon=True, fancybox=True, shadow=True,
                       title='Throughput (Left Axis)', title_fontsize=12)
    legend2 = ax2.legend(handles=line_legend_elements, loc='upper right', 
                        frameon=True, fancybox=True, shadow=True,
                        title='Throughput/Area (Right Axis)', title_fontsize=12)
    
    # 添加第一个图例回到图上
    ax.add_artist(legend1)
    
    # 调整x轴范围以更好地显示数据
    ax.set_xlim(-0.5, group_positions[-1] + 0.5)
    
    plt.tight_layout()
    return fig

def load_json_files(file_paths):
    json_data = {}
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data[os.path.basename(file_path)] = json.load(f)
    return json_data


# 使用示例数据
if __name__ == "__main__":
    # 使用方法：
    file_paths = ['vs/fusenew_analysis_results.json', 'vs/basenew_analysis_results.json', 'vs/1new_analysis_results.json', 'vs/6new_analysis_results.json']  # 您的JSON文件路径列表
    json_files = load_json_files(file_paths)
    file_data, scenarios = load_and_process_data(json_files)
    fig = create_grouped_plot(file_data, scenarios)
    plt.show()
    
    # 保存图表
    # plt.savefig('scenario_grouped_analysis.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig('scenario_grouped_analysis.png', dpi=300, bbox_inches='tight')

# 如果您有实际的JSON文件，请使用以下代码加载：


