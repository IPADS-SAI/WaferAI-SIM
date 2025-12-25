import os
import re
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 正则匹配日志行
log_pattern = re.compile(
    r"Core \d+ Rw (READ|WRITE) Time: (\d+) ns.*Address: (0x[0-9a-fA-F]+) Result: (HIT|MISS)"
)

def parse_log_file(filepath: str):
    """解析日志文件，返回 (时间, 累计HIT率) 列表"""
    data = []
    hit_count = 0
    total_count = 0

    with open(filepath, 'r') as f:
        for line in f:
            match = log_pattern.search(line)
            if not match:
                continue

            time_ns = int(match.group(2))
            result = match.group(4)

            total_count += 1
            if result == "HIT":
                hit_count += 1

            hit_rate = hit_count / total_count
            data.append((time_ns, hit_rate))

    return data


def plot_all_cores_to_single_pdf(
    input_dir="../build/gpu_cache",
    output_pdf="l1cache_hit_rate_all_in_one.pdf",
    plots_per_row=4
):
    # 匹配所有 L1Cache_cid_*.log 文件，并按 cid 数字排序
    pattern = os.path.join(input_dir, "L1Cache_cid_*.log")
    log_files = glob.glob(pattern)

    # 按 cid 的数字排序：cid_0, cid_1, ..., cid_10
    log_files.sort(key=lambda x: int(re.search(r"cid_(\d+)", os.path.basename(x)).group(1)))

    if not log_files:
        print(f"❌ 在 '{input_dir}' 目录下未找到 L1Cache_cid_*.log 文件。")
        return

    print(f"✅ 找到 {len(log_files)} 个日志文件：")
    for f in log_files:
        print(f"    {os.path.basename(f)}")

    # 计算子图布局：每行 4 个
    n_files = len(log_files)
    n_cols = plots_per_row
    n_rows = (n_files + n_cols - 1) // n_cols  # 向上取整计算行数

    # 设置图形大小：每列宽 4 英寸，每行高 3 英寸
    fig_width = n_cols * 4
    fig_height = n_rows * 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = axes.flatten() if n_files > 1 else [axes]

    # 解析并绘图
    for idx, log_file in enumerate(log_files):
        data = parse_log_file(log_file)
        if not data:
            print(f"⚠️  {os.path.basename(log_file)} 没有有效数据，跳过。")
            axes[idx].set_visible(False)
            continue

        times, hit_rates = zip(*data)
        core_id = re.search(r"cid_(\d+)", log_file).group(1)

        ax = axes[idx]
        ax.plot(times, hit_rates, linewidth=1.8)
        ax.set_title(f"Core {core_id}", fontsize=14)
        ax.set_xlabel("Time (ns)", fontsize=10)
        ax.set_ylabel("HIT Rate", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    # 隐藏多余的空子图
    for idx in range(n_files, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # 保存为单页 PDF（所有子图在一个页面上）
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\n📊 所有子图已保存至单页 PDF：'{output_pdf}'")
    print(f"   布局：{n_rows} 行 × {n_cols} 列，共 {n_files} 个子图")


if __name__ == "__main__":
    plot_all_cores_to_single_pdf()