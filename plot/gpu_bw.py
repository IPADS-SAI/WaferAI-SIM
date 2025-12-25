import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

# 读取文件
file_path = '../build/simulation_result1.txt'  # 确保路径正确
with open(file_path, 'r') as f:
    content = f.read()

content = re.sub(r'\s+', ' ', content)  # 将所有空白符压缩为空格
content = content.replace('，', ',').replace('个', '')  # 统一格式

# 正则：匹配 "以下是CACHE=..." + 下一行 [CATCH TEST] ...
pattern = r"以下是CACHE=(\d+)[^0-9]*?(\d+)\s+requests.*?\[CATCH TEST\]\s+(\d+)\s+ns[^B]*?BANDWIDTH\s+(\d+)"

matches = re.findall(pattern, content, re.DOTALL)
# 构建数据
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
        'time_ns': time_ns
    })

# 转为 DataFrame
df = pd.DataFrame(data)
print(df)

# 只保留 20 和 40 requests
df = df[df['requests'].isin([20, 40])]

# 获取唯一的 CACHE 值
cache_values = sorted(df['CACHE'].unique())

# 创建 PDF 文件
pdf_path = 'output.pdf'
with PdfPages(pdf_path) as pdf:

    # 创建子图（1行3列）
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Execution Time vs Bandwidth by Cache and Request Count", fontsize=16)

    colors = {'256': '#1f77b4', '512': '#ff7f0e', '1024': '#2ca02c'}

    for idx, cache in enumerate(cache_values):
        ax = axes[idx]
        df_cache = df[df['CACHE'] == cache]

        requests_vals = sorted(df_cache['requests'].unique())
        x = range(len(requests_vals))  # x 轴位置：20, 40
        width = 0.25
        offset = -width

        for bw in [256, 512, 1024]:
            times = []
            for req in requests_vals:
                filtered = df_cache[(df_cache['requests'] == req) & (df_cache['bandwidth'] == bw)]
                time_val = filtered['time_ns'].values[0] if len(filtered) > 0 else 0
                times.append(time_val)
            ax.bar([p + offset for p in x], times, width, label=f'Bandwidth {bw}', color=colors[str(bw)])
            offset += width

        ax.set_xlabel('Number of Requests')
        if idx == 0:
            ax.set_ylabel('Time (ns)')
        ax.set_title(f'CACHE = {cache}')
        ax.set_xticks([i for i in x])
        ax.set_xticklabels([str(r) for r in requests_vals])
        ax.grid(axis='y', alpha=0.3)

        # 第一个子图加图例
        if idx == 0:
            ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # 保存到 PDF
    pdf.savefig(fig)
    plt.close()

print(f"PDF saved to '{pdf_path}'")