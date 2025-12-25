import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


font_options = {'family':'sans-serif'}
font_legend_options = {'family':'sans-serif',
       'size':25
}
font_text_options = {'family':'sans-serif',
       'size':35
}

def parse_file(filepath):
    pattern = re.compile(
        r'\[CATCH TEST\]\s+(\d+)\s+nsHW_SRAM_SIZE\s+(\d+)\s+BANDWIDTH\s+(\d+)\s+CASE\s+(\d+)'
    )
    time_pattern = re.compile(r'花费了([\d.]+)秒')

    data = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        match = pattern.search(line.strip())
        if match:
            ns_val = int(match.group(1))
            bandwidth = int(int(match.group(3)) * 256 /4)
            case_id = int(match.group(4))
            key = (case_id, bandwidth)

            time_val = None
            j = i + 1
            while j < len(lines) and j - i <= 3:
                t_match = time_pattern.search(lines[j])
                if t_match:
                    time_val = float(t_match.group(1))
                    break
                j += 1

            if time_val is not None:
                data[key] = (ns_val, time_val)
            else:
                print(f"⚠️ 未找到耗时数据 (CASE={case_id}, BW={bandwidth}): {filepath}")
    return data

# 文件路径
file_a = '../build/simulation_result_df_pd_time_small.txt'
file_b = '../build/simulation_result_df_pd_time2_small.txt'

# 解析
data_a = parse_file(file_a)
data_b = parse_file(file_b)
print(data_a.keys(), data_b.keys())

# 共同 key
common_keys = sorted(set(data_a.keys()) & set(data_b.keys()), key=lambda x: (x[0], x[1]))
if not common_keys:
    raise ValueError("❌ 没有共同的 (CASE, BANDWIDTH) 组合！")

# 提取数据
ns_errors = []
speedups = []
labels = []
count = 0;
for key in common_keys:
    count+= 1
    a_ns, a_time = data_a[key]
    b_ns, b_time = data_b[key]

    # 以 B 为基准的相对误差: (A - B) / B * 100%
    ns_err = (a_ns - b_ns) / b_ns * 100 if b_ns != 0 else 0
    ns_errors.append(ns_err)

    # time ratio: A / B
    speedup = b_time / a_time if a_time != 0 else float('inf')
    speedups.append(speedup)

    case_id, bw = key
    labels.append(f'C{case_id}')
    if count == 6:
        break
    # labels.append(f'C{case_id}) \nBW{bw}')

# --- 绘图设置 ---
fig, ax1 = plt.subplots(figsize=(18, 8))

x = np.arange(6)  # 每个 CASE 的中心位置
width = 0.35  # 柱子宽度

color_ns = '#7E8CAD'     # 蓝色 - ns error
color_time = '#B37070'   # 红色 - time ratio

# --- 左 Y 轴：ns 误差 (%)
bars1 = ax1.bar(x - width/2 - 0.1, ns_errors, width, color=color_ns, alpha=0.85,
                edgecolor='black', linewidth=0.6, hatch='\/\/',
                label='Simulation Time(ns) Error (%)')

ax1.set_ylabel('Relative Error (%)', fontsize=25, fontdict=font_options)
# ax1.set_xlabel('Test Case (CASE ID, BANDWIDTH)', fontsize=12)
ax1.tick_params(axis='y', labelsize=20)
ax1.tick_params(axis='x', labelsize=20)
ax1.axhline(0, color='gray', linewidth=1.0, linestyle='-', alpha=0.7)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

# 左轴：正负误差独立扩展
if ns_errors:
    upper = max(ns_errors)
    lower = min(ns_errors)
    y1_top = upper * 4.3 if upper > 0 else upper * 0.7
    y1_bottom = lower * 1.3 if lower < 0 else lower * 0.7
    if abs(y1_top - y1_bottom) < 1:
        center = (upper + lower) / 2
        y1_top, y1_bottom = center + 0.6, center - 0.6
else:
    y1_top, y1_bottom = 1, -1
ax1.set_ylim(bottom=y1_bottom, top=y1_top)



# 添加 ns 误差标签
for bar, val in zip(bars1, ns_errors):
    height = bar.get_height()
    va = 'bottom' if height >= 0 else 'top'
    y_text = height + (0.015 * max([abs(v) for v in ns_errors] + [1])) * (1 if height >= 0 else -1)
    ax1.text(bar.get_x() + bar.get_width()/2., y_text, f'{val:+.2f}%',
             ha='center', va=va, fontsize=14, fontweight='bold', color=color_ns)

# --- 右 Y 轴：time ratio (A/B)
ax2 = ax1.twinx()
# 右轴：时间比，只扩展顶部
max_speedup = max(speedups) if speedups else 1
ax2.set_ylim(bottom=0, top=max_speedup * 1.3)
bars2 = ax2.bar(x + width/2 + 0.1, speedups, width, color=color_time, alpha=0.75,
                edgecolor='darkred', linewidth=0.8, hatch='//',
                label='Time Ratio (Perf/TLM)')

ax2.set_ylabel('Simulation time speed-up ratio', fontsize=25)
ax2.tick_params(axis='y',  labelsize=20)
ax2.set_ylim(bottom=0)  # 时间比 ≥ 0

# 添加 time ratio 标签
for bar, val in zip(bars2, speedups):
    height = bar.get_height()
    # y_text = height * 1.05 if height > 0 else 0.1
    y_text = height + 0.05 if height > 0 else 0.1
    ax2.text(bar.get_x() + bar.get_width()/2., y_text, f'{val:.2f}×',
             ha='center', va='bottom', fontsize=14, fontweight='bold', color=color_time)
    


# # --- 标题与图例 ---
# plt.title('Comparison: A vs B\nBar1 (left): [CATCH TEST] ns Error % (A vs B)\nBar2 (right): Time Ratio A/B (×)',
#           fontsize=14, fontweight='bold', pad=20)

# 合并图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=color_ns, edgecolor='black', alpha=0.85,
          label='Simulation Time(ns) Error (%)'),
    Patch(facecolor=color_time, edgecolor='darkred', alpha=0.75, hatch='//',
          label='Time Ratio Perf/TLM ')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=16)

# 布局优化
fig.tight_layout()

# 保存为 PDF
output_pdf = 'comparison_dual_axis_separated.pdf'
with PdfPages(output_pdf) as pdf:
    pdf.savefig(fig)
    plt.close()

print(f"✅ 图表已保存为: {output_pdf}")