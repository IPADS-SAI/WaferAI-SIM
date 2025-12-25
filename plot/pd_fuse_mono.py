import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict

# 设置学术风格的字体和参数
plt.rcParams.update({
    'font.size': 10,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.edgecolor': 'black',
})

def parse_data_file(file_path):
    """解析数据文件，通过文件名中的唯一标记分组"""
    configs_by_group = defaultdict(dict)
    current_config = None
    current_request = None
    current_group = None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('*') and line.endswith('*'):
            # 提取配置名称和唯一标记
            # 格式: *../llm/test/workload_config/gpt2_small/pd_split/pd_split_49_14_100_100.json*
            config_match = re.search(r'pd_split_(\d+)_(\d+)_(\d+_\d+)', line)
            if not config_match:
                # 尝试其他可能的格式
                config_match = re.search(r'pd_split_(\d+)_(\d+)_([^./\s*]+)', line)
            
            if config_match:
                p_cores = int(config_match.group(1))
                d_cores = int(config_match.group(2))
                group_id = config_match.group(3)  # 唯一标记，如 "100_100"
                
                current_config = f"{p_cores}_{d_cores}"
                current_group = group_id
                
                # 初始化配置
                if current_config not in configs_by_group[current_group]:
                    configs_by_group[current_group][current_config] = {
                        'p_cores': p_cores,
                        'd_cores': d_cores,
                        'requests': [],
                        'group_id': group_id
                    }
                
                print(f"Found config: P{p_cores}/D{d_cores} in group '{group_id}'")
        
        elif line.startswith('Request'):
            request_match = re.search(r'Request (\d+):', line)
            if request_match and current_config and current_group:
                current_request = int(request_match.group(1))
                configs_by_group[current_group][current_config]['requests'].append([])
        
        elif line.startswith('Token'):
            token_match = re.search(r'Token (\d+): ([\d.]+)', line)
            if token_match and current_config and current_request is not None and current_group:
                token_idx = int(token_match.group(1))
                timestamp = float(token_match.group(2))
                configs_by_group[current_group][current_config]['requests'][current_request].append(timestamp)
    
    # 统计每个组的信息
    print(f"\nFound {len(configs_by_group)} experiment groups:")
    for group_id, configs in configs_by_group.items():
        # 计算该组的平均token数
        all_token_counts = []
        for config_name, config_data in configs.items():
            for request_tokens in config_data['requests']:
                if len(request_tokens) > 0:
                    all_token_counts.append(len(request_tokens))
        
        if all_token_counts:
            median_tokens = int(np.median(all_token_counts))
            print(f"  Group '{group_id}': {len(configs)} configurations, ~{median_tokens} tokens per request")
    
    return configs_by_group

def calculate_metrics(configs):
    """计算性能指标"""
    metrics = {}
    
    for config_name, config_data in configs.items():
        p_cores = config_data['p_cores']
        d_cores = config_data['d_cores']
        requests = config_data['requests']
        
        ttft_multiplier = (49 / p_cores - 1) / 2 + 1
        
        # 计算TTFT (Time to First Token)
        ttfts = []
        # 计算Time Between Tokens
        tbts = []
        # 计算总延迟
        total_latencies = []
        
        # 获取所有请求的开始时间
        all_start_times = []
        for request_tokens in requests:
            if len(request_tokens) > 0:
                all_start_times.append(request_tokens[0])
        
        earliest_start = 0
        
        for request_tokens in requests:
            if len(request_tokens) >= 2:  # 确保有足够的token
                ttft_original = (request_tokens[0] - earliest_start) / 1e9
                ttft_adjusted = ttft_original * ttft_multiplier
                ttfts.append(ttft_adjusted)
                
                # TBT
                if len(request_tokens) > 1:
                    decode_times = []
                    for i in range(len(request_tokens)-1):
                        decode_times.append((request_tokens[i+1] - request_tokens[i]) / 1e9)
                    if decode_times:
                        tbts.append(np.min(decode_times))
                
                # 总延迟
                total_latency = (request_tokens[-1] - 0) / 1e9
                total_latencies.append(total_latency)
        
        # 计算吞吐量
        if requests and all_start_times:
            total_tokens = sum(len(req) for req in requests if len(req) > 0)
            all_end_times = []
            for request_tokens in requests:
                if len(request_tokens) > 0:
                    all_end_times.append(request_tokens[-1])
            
            if all_end_times:
                latest_end = max(all_end_times)
                total_time = (latest_end - earliest_start) / 1e9
                throughput = total_tokens / total_time if total_time > 0 else 0
            else:
                throughput = 0
        else:
            throughput = 0
        
        # 计算实际的token数统计
        token_counts = [len(req) for req in requests if len(req) > 0]
        avg_tokens = np.mean(token_counts) if token_counts else 0
        
        metrics[config_name] = {
            'p_cores': p_cores,
            'd_cores': d_cores,
            'ttft_mean': np.mean(ttfts) if ttfts else 0, 
            'ttft_std': np.std(ttfts) if ttfts else 0,   
            'ttft_multiplier': ttft_multiplier,          
            'tbt_mean': np.mean(tbts) if tbts else 0,
            'tbt_std': np.std(tbts) if tbts else 0,
            'latency_mean': np.mean(total_latencies) if total_latencies else 0,
            'latency_std': np.std(total_latencies) if total_latencies else 0,
            'throughput': throughput,
            'total_cores': p_cores + d_cores,
            'num_requests': len([r for r in requests if len(r) > 0]),
            'avg_tokens': avg_tokens
        }
    
    return metrics

def get_string_ratio(input, output):
    # if input == output:
    #     return "1:1 (" + input + " tokens)"
    # input = int(input)
    # output = int(output)
    # if input > output:
    #     input /= output
    #     return str(int(input)) + ":1"
    # else:
    #     output /= input
    #     return "1:" + str(int(output))
    return input + ":" + output

def calculate_global_y_ranges(metrics_by_group):
    """计算所有组中的最大值，用于统一y轴刻度"""
    all_latencies = []
    all_ttfts = []
    all_tbts = []
    
    for group_id, metrics in metrics_by_group.items():
        for config_name, data in metrics.items():
            all_latencies.append(data['latency_mean'])
            all_ttfts.append(data['ttft_mean'])
            all_tbts.append(data['tbt_mean'])
    
    # 计算范围，添加10%的余量
    max_latency = max(all_latencies) * 1.1 if all_latencies else 0
    max_ttft_tbt = max(max(all_ttfts) if all_ttfts else [0], 
                       max(all_tbts) if all_tbts else [0]) * 1.1
    
    return max_latency, max_ttft_tbt

def plot_experiment_group(ax1, metrics, group_id, subplot_idx, total_subplots, rows, cols, 
                         global_latency_max, global_ttft_tbt_max):
    """为单个实验组绘制图表，使用统一的y轴刻度"""
    # 按P核数量排序（P核多的在前）
    sorted_configs = sorted(metrics.items(), key=lambda x: (-x[1]['p_cores'], x[1]['d_cores']))
    config_names = [item[0] for item in sorted_configs]
    config_data = [item[1] for item in sorted_configs]
    
    # 提取数据
    p_cores = [data['p_cores'] for data in config_data]
    d_cores = [data['d_cores'] for data in config_data]
    ttft = [data['ttft_mean'] for data in config_data]  # 已经应用了倍率
    ttft_multipliers = [data['ttft_multiplier'] for data in config_data]
    tbt = [data['tbt_mean'] for data in config_data]
    latency = [data['latency_mean'] for data in config_data]
    throughput = [data['throughput'] for data in config_data]
    num_requests = [data['num_requests'] for data in config_data]
    avg_tokens = [data['avg_tokens'] for data in config_data]
    
    # 设置x轴
    x_pos = np.arange(len(config_names))
    config_labels = [f"P{p}/D{d}" for p, d in zip(p_cores, d_cores)]
    
    # 颜色设置
    color1 = '#EE4266'  # 蓝色 - TTFT
    color2 = '#7E8CAD'  # 红色 - 延迟
    color4 = '#1A8C5C'  # 橙色 - TBT
    
    # 绘制平均延迟 (柱状图)
    bars = ax1.bar(x_pos, latency, 0.4, label='Avg. Latency', 
                   color=color2, alpha=0.8, hatch="//", edgecolor='black', linewidth=0.8)
    
    # 判断是否显示左侧y轴标签（只有最左侧列显示）
    current_col = subplot_idx % cols
    # if current_col == 0:  # 最左侧列
    #     ax1.set_ylabel('Avg. Latency (ms)', fontsize=22, fontweight='bold')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(config_labels, fontsize=18, rotation=30)
    current_row = subplot_idx // cols
    if current_row != rows - 1:  # 不是最后一行
        ax1.tick_params(axis='x', bottom=True, labelbottom=False)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax1.yaxis.get_offset_text().set_fontsize(0)
    ax1.tick_params(axis='y', labelsize=25)
    ax1.tick_params(axis='x', labelsize=20)

    # 设置统一的左y轴范围
    ax1.set_ylim(0, global_latency_max)
    if current_col != 0:
        ax1.tick_params(axis='y', left=False, labelleft=False)
    
    # 右y轴 - TTFT和TBT
    ax2 = ax1.twinx()
    
    # 绘制TTFT和TBT
    line1 = ax2.plot(x_pos, ttft, 'o-', color=color1, linewidth=2, markersize=6, 
                     label='TTFT', markerfacecolor='white', markeredgewidth=1.5)
    
    line2 = ax2.plot(x_pos, tbt, '^-', color=color4, linewidth=2, markersize=6,
                     label='TBT (ms)', markerfacecolor='white', markeredgewidth=1.5)
    
    # 判断是否显示右侧y轴标签（只有最右侧列显示）
    # if current_col == cols - 1:  # 最右侧列
    #     ax2.set_ylabel('TTFT & TBT (ms)', color='black', fontsize=22, fontweight='bold')
    
    ax2.tick_params(axis='y', labelcolor='black', labelsize=25)
    ax2.tick_params(axis='x', labelsize=14)
    
    # 设置统一的右y轴范围
    ax2.set_ylim(0, global_ttft_tbt_max)
    if current_col != cols - 1:
        ax2.tick_params(axis='y', right=False, labelright=False)
    
    # 添加网格
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # 添加标题（显示组ID和平均token数）
    if avg_tokens:
        avg_token_count = int(np.mean(avg_tokens))
        avg_request_count = int(np.mean(num_requests))
        
        # 解析group_id为输入输出token数
        group_parts = group_id.split('_')
        if len(group_parts) == 2:
            input_tokens = group_parts[0]
            output_tokens = group_parts[1]
            title_text = f'P:D = {get_string_ratio(input_tokens, output_tokens)}'
        else:
            # 如果格式不符合预期，使用原始group_id
            title_text = f'Group: {group_id}\n(~{avg_token_count - 1} tokens/req, {avg_request_count} reqs/config)'
        
        ax1.set_title(title_text, fontsize=25, fontweight='bold', pad=10)
    else:
        # 尝试解析group_id
        group_parts = group_id.split('_')
        if len(group_parts) == 2:
            input_tokens = group_parts[0]
            output_tokens = group_parts[1]
            title_text = f'Prefill:Decode = {get_string_ratio(input_tokens, output_tokens)}'
        else:
            title_text = f'Group: {group_id}'
        
        ax1.set_title(title_text, fontsize=11, fontweight='bold', pad=10)
    
    # 打印统计信息
    print(f"\nGroup '{group_id}' Performance:")
    for i, label in enumerate(config_labels):
        print(f"  {label}: Latency={latency[i]:.2f}s, TTFT={ttft[i]:.3f}s, "
              f"TBT={tbt[i]:.3f}s, Throughput={throughput[i]:.1f} tokens/s, "
              f"Tokens={avg_tokens[i]:.0f}, Requests={num_requests[i]}")
    
    # 返回图例元素（用于创建统一图例）
    if subplot_idx == 0:
        return ax1, ax2
    return None, None

# 主程序
print("Parsing data file...")
print("-" * 70)

# 解析数据文件
configs_by_group = parse_data_file('pd_fuse_mono.txt')

# 计算每组实验的指标
metrics_by_group = {}
for group_id, configs in configs_by_group.items():
    metrics_by_group[group_id] = calculate_metrics(configs)

# 确定子图数量和布局
num_experiments = len(metrics_by_group)
if num_experiments == 0:
    print("No data found!")
    exit()

# 计算全局y轴范围
global_latency_max, global_ttft_tbt_max = calculate_global_y_ranges(metrics_by_group)
print(f"\nGlobal Y-axis ranges:")
print(f"  Left axis (Latency): 0 - {global_latency_max:.4f}")
print(f"  Right axis (TTFT/TBT): 0 - {global_ttft_tbt_max:.4f}")

# 计算子图布局
if num_experiments <= 2:
    rows, cols = 1, num_experiments
elif num_experiments <= 4:
    rows, cols = 2, 2
elif num_experiments <= 6:
    rows, cols = 2, 3
elif num_experiments <= 9:
    rows, cols = 3, 3
else:
    rows = (num_experiments + 3) // 4
    cols = 4

# 创建图表
fig_width = 5 * cols
fig_height = 4 * rows
fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

# 如果只有一个子图，确保axes是列表格式
if num_experiments == 1:
    axes = [axes]
elif rows == 1 and cols == 1:
    axes = [axes]
elif rows == 1 or cols == 1:
    axes = axes
else:
    axes = axes.flatten()

# 按group_id排序（可以自定义排序逻辑）
sorted_groups = sorted(metrics_by_group.keys())

# 为每组实验创建子图
first_ax1, first_ax2 = None, None
for idx, group_id in enumerate(sorted_groups):
    if idx < len(axes):
        ax1, ax2 = plot_experiment_group(axes[idx], metrics_by_group[group_id], 
                                         group_id, idx, num_experiments, rows, cols,
                                         global_latency_max, global_ttft_tbt_max)
        if idx == 0:
            first_ax1, first_ax2 = ax1, ax2

# 隐藏多余的子图
if num_experiments < len(axes):
    for idx in range(num_experiments, len(axes)):
        fig.delaxes(axes[idx])

# 添加统一的图例（在所有子图外面）
if first_ax1 and first_ax2:
    lines1, labels1 = first_ax1.get_legend_handles_labels()
    lines2, labels2 = first_ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper center', 
              bbox_to_anchor=(0.5, 0.97),
              ncol=3,  # 水平排列
              fontsize=25, 
              frameon=True,
              edgecolor='black',
              fancybox=False,
              shadow=False)
    


# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.8, hspace=0.35, wspace=0.1, left=0.12, right=0.88)

# 在图的外部统一加 y 轴标签（居中）
fig.text(0.04, 0.5, 'Avg. Latency (×10⁴ms)', va='center', rotation='vertical',
         fontsize=30, fontweight='bold')
fig.text(0.96, 0.5, 'TTFT & TBT (ms)', va='center', rotation='vertical',
         fontsize=30, fontweight='bold')

# 显示图表
# 保存图表
# plt.savefig('pd_core_allocation_multi_experiments.png', dpi=300, bbox_inches='tight')
plt.savefig('pd_core_allocation_multi_experiments.pdf', dpi=300, bbox_inches='tight')

print("\nFigures saved as 'pd_core_allocation_multi_experiments.png' and '.pdf'")

# 打印倍率信息总结
print("\n" + "=" * 70)
print("TTFT Multiplier Summary:")
print("-" * 70)