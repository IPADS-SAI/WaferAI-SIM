import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import math

def parse_file(file_path, source_label):
    with open(file_path, 'r') as f:
        content = f.read()

    # 标准化空白符和标点
    content = re.sub(r'\s+', ' ', content)
    content = content.replace('，', ',').replace('个', '')

    pattern = r"以下是CACHE=(\d+)[^0-9]*?(\d+)\s+requests.*?\[CATCH TEST\]\s+(\d+)\s+ns[^B]*?BANDWIDTH\s+(\d+)"
    matches = re.findall(pattern, content, re.DOTALL)

    data = []
    for match in matches:
        cache = int(match[0])
        requests = int(match[1])
        time_ns = int(match[2])
        bandwidth = int(match[3])
        data.append({
            'CACHE': cache,
            'requests': requests,
            'bandwidth': bandwidth,
            'time_ns': time_ns,
            'source': source_label
        })
    return data


# === 读取两个文件 ===
data1 = parse_file('../build/simulation_result1.txt', 'L1/L2 Cache')
data2 = parse_file('../build/simulation_result_df_pd1.txt', 'Scratchpad')

# 合并
df = pd.DataFrame(data1 + data2)

# 只保留 requests = 20, 40
df = df[df['requests'].isin([20, 40])]

# 排序
df['requests'] = pd.Categorical(df['requests'], categories=[20, 40], ordered=True)
df['bandwidth'] = pd.Categorical(df['bandwidth'], categories=[256, 512, 1024], ordered=True)

# 获取唯一的 CACHE 值
cache_values = sorted(df['CACHE'].unique())

# === 颜色设置 ===
colors = {'256': '#7E8CAD', '512': '#B37070', '1024': '#7A9273'}
hatches = {'L1/L2 Cache': '...', 'Scratchpad': '///'}
alpha = 0.8

# === 计算全局 y 轴最大值 ===
max_time = df['time_ns'].max() + 5e7  # 所有时间中的最大值
# 向上取整到最近的 1e8 或 5e7 的倍数，便于显示
def round_up_to_n(x, n):
    return int(math.ceil(x / n)) * n

if max_time < 1e8:
    max_time_rounded = round_up_to_n(max_time, 1e7)
elif max_time < 5e8:
    max_time_rounded = round_up_to_n(max_time, 5e7)
else:
    max_time_rounded = round_up_to_n(max_time, 1e8)

# === 生成 PDF 图表 ===
pdf_path = 'combined_output_uniform_y.pdf'
with PdfPages(pdf_path) as pdf:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # fig.suptitle("Execution Time Comparison: L1/L2 Cache vs Scratchpad\n(CACHE Size Variation)", fontsize=14)

    for idx, cache in enumerate(cache_values):
        ax = axes[idx]
        df_cache = df[df['CACHE'] == cache]

        requests_vals = sorted(df_cache['requests'].unique())
        x = range(len(requests_vals))  # x 轴：20, 40
        width = 0.1
        bw_offset = 0.4
        source_offset_base = {'L1/L2 Cache': -bw_offset/2, 'Scratchpad': bw_offset/2}

        for bw in [256, 512, 1024]:
            for source in ['L1/L2 Cache', 'Scratchpad']:
                times = []
                for req in requests_vals:
                    filtered = df_cache[
                        (df_cache['requests'] == req) &
                        (df_cache['bandwidth'] == bw) &
                        (df_cache['source'] == source)
                    ]
                    time_val = filtered['time_ns'].values[0] if len(filtered) > 0 else 0
                    times.append(time_val)

                # 计算位置：使用对数分布错开 bandwidth
                positions = [
                    i + 0.8 * (math.log2(bw) - math.log2(512)) / math.log2(256) + source_offset_base[source]
                    for i in x
                ]
                ax.bar(positions, times, width=width, label=f'{source}, BW={bw}',
                       color=colors[str(bw)], alpha=alpha, hatch=hatches[source], edgecolor='black')

        ax.set_xlabel('Number of Requests', fontsize=20)
        if idx == 0:
            ax.set_ylabel('Time (ns)', fontsize=20)
        ax.set_title(f'CACHE/SCRATCHPAD = {cache} MB', fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in requests_vals])
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(axis='y', alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        ax.yaxis.get_offset_text().set_fontsize(15)

        # ✅ 设置统一的 y 轴范围
        ax.set_ylim(0, max_time_rounded)

    # 图例去重
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.8), ncol=6 , fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

    pdf.savefig(fig)
    plt.close()

print(f"Combined PDF with uniform y-axis saved to '{pdf_path}'")