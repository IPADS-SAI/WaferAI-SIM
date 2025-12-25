import os
import re
import matplotlib.pyplot as plt
import numpy as np
import math

def parse_log_file(filepath):
    """
    Parse a log file and extract time and ratio data
    """
    times = []
    ratios = []
    
    # Regular expression to match log entries
    pattern = r"Time:\s*([\d.]+)\s*ns\s+Free/Total Ratio:\s*([\d.]+)"
    
    with open(filepath, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                time = float(match.group(1))
                ratio = float(match.group(2))
                times.append(time)
                ratios.append(ratio)
    
    return times, ratios

def plot_sram_utilization():
    """
    Plot SRAM utilization ratios from all log files in subplots and save as PDF
    """
    # Directory containing log files
    log_dir = "../build/sram_util/"
    
    # Check if directory exists
    if not os.path.exists(log_dir):
        print(f"Directory {log_dir} does not exist")
        return
    
    # Find all log files matching the pattern
    log_files = [f for f in os.listdir(log_dir) if re.match(r'sram_manager_cid_\d+\.log', f)]
    
    if not log_files:
        print("No log files found matching the pattern 'sram_manager_cid_*.log'")
        return
    
    # Calculate subplot layout
    n_files = len(log_files)
    n_cols = min(3, n_files)  # Maximum 3 columns
    n_rows = math.ceil(n_files / n_cols)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Handle case when there's only one subplot
    if n_files == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Process each log file
    for i, log_file in enumerate(log_files):
        filepath = os.path.join(log_dir, log_file)
        
        # Parse the log file
        times, ratios = parse_log_file(filepath)
        
        # Extract CID from filename
        cid_match = re.search(r'sram_manager_cid_(\d+)\.log', log_file)
        cid = cid_match.group(1) if cid_match else "Unknown"
        
        # Plot the data in the corresponding subplot
        axes[i].plot(times, ratios, marker='o', linewidth=2, markersize=4, color=f'C{i}')
        axes[i].set_title(f'CID {cid}', fontsize=12)
        axes[i].set_xlabel('Time (ns)', fontsize=10)
        axes[i].set_ylabel('Free/Total Ratio', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # Set y-axis limits to 0-1 for all subplots
        axes[i].set_ylim(0, 1)
        
        # Format x-axis to show scientific notation if needed
        axes[i].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Hide any unused subplots
    for i in range(n_files, len(axes)):
        fig.delaxes(axes[i])
    
    # Add main title
    fig.suptitle('SRAM Utilization Over Time by CID', fontsize=14)
    
    # Adjust layout and save as PDF
    plt.tight_layout()
    output_file = "sram_utilization.pdf"
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    plot_sram_utilization()