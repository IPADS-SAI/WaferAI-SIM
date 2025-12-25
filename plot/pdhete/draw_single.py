import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_palette("husl")

# CHIP_AREA = [749.4, 761.7, 798.5, 685.29, 758.97, 861.45, 648.45, 636.17, 623.89, 726.37, 738.65, 714.09]
# CHIP_AREA = [726.37, 749.4, 714.09, 761.7, 738.65, 798.5, 861.45, 623.89, 636.17, 648.45, 685.29, 758.97]
CHIP_AREA = [726.37, 749.4, 761.7, 798.5, 861.45, 623.89, 636.17, 648.45, 685.29, 758.97]
x_label = [ 'base_A64H60', '1_A64H30', '2_A64H120', '3_A64H240', '4_A64H480', '5_A32H30', '6_A32H60', '7_A32H120', '8_A32H240', '9_A32H480']

def load_and_process_data_multiple(json_file_paths):
    """Load multiple JSON files, only first two configs per file"""
    all_data = []
    cnt = 0

    for file_path in json_file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)

        configs = []
        tbt_mins = []
        throughput_vals = []
        throughput_per_area = []
        tbt_per_area = []

        # Only take first two items
        for config_name, metrics in list(data.items())[:2]:
            configs.append(config_name)
            tbt_mins.append(metrics['tbt']['min'] / 1e12)    # sec
            throughput_vals.append(metrics['throughput_tokens_per_second']*1000)
            throughput_per_area.append(
                1000*metrics['throughput_tokens_per_second'] / CHIP_AREA[cnt]
            )
            tbt_per_area.append(
                (metrics['tbt']['mean'] / 1e12) / CHIP_AREA[cnt]
            )

        all_data.append({
            'file': file_path,
            'configs': configs,
            'tbt': tbt_mins,
            'throughput': throughput_vals,
            'throughput_area': throughput_per_area,
            'tbt_area': tbt_per_area
        })

        cnt += 1

    return all_data


def create_performance_visualization_multiple(json_file_paths):
    """Plot dual subplots: Left (Throughput + Throughput/Area), Right (TBT + TBT/Area)"""
    all_data = load_and_process_data_multiple(json_file_paths)

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(24, 6))

    num_files = len(json_file_paths)
    x_pos = np.arange(num_files)
    bar_width = 0.35

    colors = ['#7A9273', '#CC5500']  # cfg1, cfg2 用不同颜色

    # ======================
    # LEFT SUBPLOT: Throughput
    # ======================
    throughput_cfg1 = [d['throughput'][0]/1000 for d in all_data]
    throughput_cfg2 = [d['throughput'][1]/1000 for d in all_data]
    
    print(throughput_cfg1)

    ax1.bar(x_pos - bar_width/2, throughput_cfg1, bar_width, edgecolor='black', linewidth=0.5, alpha=0.85,
            label="Throughput - Output 100", hatch = "//", color='#B37070')
    ax1.bar(x_pos + bar_width/2, throughput_cfg2, bar_width, edgecolor='black', linewidth=0.5, alpha=0.85,
            label="Throughput - Output 200",hatch = "--", color='#7A9273')

    ax1.set_ylabel('Throughput\n(×10³tokens/s)', fontsize=28, fontweight='bold')
    ax1.set_title('(a) Throughput', fontsize=30, fontweight='bold')

    # 横坐标标签
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_label, fontsize=24, rotation=30, ha='right')

    # Secondary axis for Throughput/Area
    ax2 = ax1.twinx()
    ax2.set_ylabel('Throughput/Area\n(tokens/s/mm²)', fontsize=28, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=24)
    ax2.tick_params(axis='y', labelsize=24)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax2.yaxis.get_offset_text().set_fontsize(24)
    ax1.set_ylim(0, max(throughput_cfg1) * 1.3)
    # 两根折线（Throughput/Area）
    tpa_cfg1 = [d['throughput_area'][0] for d in all_data]
    tpa_cfg2 = [d['throughput_area'][1] for d in all_data]
    print("ea")
    print(tpa_cfg1)

    ax2.plot(x_pos, tpa_cfg1, 'o-', linewidth=3, markersize=8, 
             label="Throughput/Area - Output 100", color='#FF7F0E')
    ax2.plot(x_pos, tpa_cfg2, 's-', linewidth=3, markersize=8, 
             label="Throughput/Area - Output 200", color='#1a8c5c')
    ax2.set_ylim(0, max(tpa_cfg1) * 1.3)

    # Legends for left subplot
    ax1.legend(loc='upper left', fontsize=15, frameon=True, fancybox=False, shadow=False)
    ax2.legend(loc='upper right', fontsize=15, frameon=True, fancybox=False, shadow=False)

    # ======================
    # RIGHT SUBPLOT: TBT
    # ======================
    tbt_cfg1 = [d['tbt'][0] for d in all_data]
    tbt_cfg2 = [d['tbt'][1] for d in all_data]

    ax3.bar(x_pos - bar_width/2, tbt_cfg1, bar_width, edgecolor='black', linewidth=0.5, alpha=0.85,
            label="TBT - Output 100",hatch = "//", color='#B37070')
    ax3.bar(x_pos + bar_width/2, tbt_cfg2, bar_width, edgecolor='black', linewidth=0.5, alpha=0.85,
            label="TBT - Output 200", hatch = "--", color='#7A9273')

    ax3.set_ylabel('TBT (s)', fontsize=28, fontweight='bold')
    ax3.set_title('(b) Time between tokens', fontsize=30, fontweight='bold')

    # 横坐标标签
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_label, fontsize=24, rotation=30, ha='right')

    # Secondary axis for TBT/Area
    ax4 = ax3.twinx()
    ax4.set_ylabel('TBT/Area (s/mm²)', fontsize=28, fontweight='bold')
    ax3.tick_params(axis='y', labelsize=24)
    ax4.tick_params(axis='y', labelsize=24)
    ax3.set_ylim(0, max(tbt_cfg2) * 1.3)


    # 两根折线（TBT/Area）
    tbt_area_cfg1 = [d['tbt_area'][0] for d in all_data]
    tbt_area_cfg2 = [d['tbt_area'][1] for d in all_data]
    print("aaa")
    print(tbt_area_cfg1)

    ax4.plot(x_pos, tbt_area_cfg1, 'o-', linewidth=3, markersize=8, 
             label="TBT/Area - Output 100", color='#FF7F0E')
    ax4.plot(x_pos, tbt_area_cfg2, 's-', linewidth=3, markersize=8, 
             label="TBT/Area - Output 200", color='#1a8c5c')
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax4.yaxis.get_offset_text().set_fontsize(15)

    # Legends for right subplot
    ax3.legend(loc='upper left', fontsize=15, frameon=True, fancybox=False, shadow=False)
    ax4.legend(loc='upper right', fontsize=15, frameon=True, fancybox=False, shadow=False)
    ax4.set_ylim(0, max(tbt_area_cfg2) * 1.3)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    plt.savefig('llm_performance_analysis_dual.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig('llm_performance_analysis_dual.png', dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    files = [
        "base_analysis_results.json",
        "9_analysis_results.json",
        # "pdhete/11_analysis_results.json",
        "1_analysis_results.json",
        # "pdhete/10_analysis_results.json",
        "2_analysis_results.json",
        "5_analysis_results.json",
        "8_analysis_results.json",
        "7_analysis_results.json",
        "6_analysis_results.json",
        "3_analysis_results.json",
        "4_analysis_results.json",
    ]
    create_performance_visualization_multiple(files)