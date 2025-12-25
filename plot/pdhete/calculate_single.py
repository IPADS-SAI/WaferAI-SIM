import re
import statistics
from pathlib import Path
import csv
import json

def parse_data_file(file_path):
    """
    解析包含时间数据的文本文件
    
    Args:
        file_path (str): 输入文件路径
        
    Returns:
        dict: 解析后的数据，按组织结构组织
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式分割数据组
    groups = re.split(r'\*([^*]+)\*', content)
    
    parsed_data = {}
    
    for i in range(1, len(groups), 2):
        if i + 1 >= len(groups):
            break
            
        group_title = groups[i].strip()
        group_content = groups[i + 1].strip()
        
        if not group_content:
            continue
            
        requests = parse_requests(group_content)
        if requests:
            parsed_data[group_title] = requests
    
    return parsed_data

def parse_requests(content):
    """
    解析单个组内的请求数据
    
    Args:
        content (str): 组内容文本
        
    Returns:
        list: 请求数据列表
    """
    requests = []
    
    # 按请求分割
    request_sections = re.split(r'Request (\d+):', content)
    
    for i in range(1, len(request_sections), 2):
        if i + 1 >= len(request_sections):
            break
            
        request_id = int(request_sections[i])
        request_content = request_sections[i + 1].strip()
        
        # 提取token时间
        token_times = []
        token_pattern = r'Token (\d+): ([\d.]+)'
        matches = re.findall(token_pattern, request_content)
        
        for token_id, time_str in matches:
            token_times.append({
                'token_id': int(token_id),
                'time': float(time_str)
            })
        
        if token_times:
            requests.append({
                'request_id': request_id,
                'tokens': token_times
            })
    
    return requests

def calculate_metrics(group_data):
    """
    计算TTFT、TBT、平均latency和throughput
    
    Args:
        group_data (list): 单组的请求数据
        
    Returns:
        dict: 计算结果
    """
    if not group_data:
        return None
    
    ttft_times = []  # Time to First Token
    tbt_times = []   # Time Between Tokens
    latency_times = []  # Total latency (time to last token)
    all_last_token_times = []  # 所有请求的最后一个token时间
    total_tokens = 0  # 总token数量
    
    for request in group_data:
        tokens = request['tokens']
        if not tokens:
            continue
            
        # 按token_id排序确保顺序正确
        tokens = sorted(tokens, key=lambda x: x['token_id'])
        
        # 统计token数量
        total_tokens += len(tokens)
        
        # TTFT: 第0个token的时间（从开始到第一个token）
        if tokens:
            ttft = tokens[0]['time']
            ttft_times.append(ttft)
        
        # Latency: 最后一个token的时间
        if tokens:
            latency = tokens[-1]['time']
            latency_times.append(latency)
            all_last_token_times.append(latency)
        
        # TBT: 相邻token之间的时间差
        for i in range(1, len(tokens)):
            tbt = tokens[i]['time'] - tokens[i-1]['time']
            tbt_times.append(tbt)
    
    # 计算throughput: 最晚完成的请求时间 / 总token数
    max_completion_time = max(all_last_token_times) if all_last_token_times else 0
    throughput_tokens_per_ns = total_tokens / statistics.mean(latency_times) 
    throughput_tokens_per_second = throughput_tokens_per_ns * 1e9  # 转换为每秒token数
    
    # 计算统计指标
    results = {
        'request_count': len(group_data),
        'token_count_per_request': len(group_data[0]['tokens']) if group_data else 0,
        'total_tokens': total_tokens,
        'max_completion_time_ns': max_completion_time,
        'throughput_tokens_per_second': throughput_tokens_per_second,
        'ttft': {
            'mean': statistics.mean(ttft_times) if ttft_times else 0,
            'median': statistics.median(ttft_times) if ttft_times else 0,
            'min': min(ttft_times) if ttft_times else 0,
            'max': max(ttft_times) if ttft_times else 0,
            'std': statistics.stdev(ttft_times) if len(ttft_times) > 1 else 0
        },
        'tbt': {
            'mean': statistics.mean(tbt_times) if tbt_times else 0,
            'median': statistics.median(tbt_times) if tbt_times else 0,
            'min': min(tbt_times) if tbt_times else 0,
            'max': max(tbt_times) if tbt_times else 0,
            'std': statistics.stdev(tbt_times) if len(tbt_times) > 1 else 0
        },
        'latency': {
            'mean': statistics.mean(latency_times) if latency_times else 0,
            'median': statistics.median(latency_times) if latency_times else 0,
            'min': min(latency_times) if latency_times else 0,
            'max': max(latency_times) if latency_times else 0,
            'std': statistics.stdev(latency_times) if len(latency_times) > 1 else 0
        }
    }
    
    return results

def format_time(nanoseconds):
    """
    将纳秒转换为更易读的格式
    """
    if nanoseconds >= 1e9:
        return f"{nanoseconds / 1e9:.3f} s"
    elif nanoseconds >= 1e6:
        return f"{nanoseconds / 1e6:.3f} ms"
    elif nanoseconds >= 1e3:
        return f"{nanoseconds / 1e3:.3f} μs"
    else:
        return f"{nanoseconds:.3f} ns"

def save_results_to_csv(results, output_path):
    """
    将结果保存为CSV文件
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'Group', 'Request_Count', 'Token_Count_Per_Request', 'Total_Tokens', 
            'Max_Completion_Time_ns', 'Throughput_Tokens_Per_Second',
            'TTFT_Mean_ns', 'TTFT_Median_ns', 'TTFT_Min_ns', 'TTFT_Max_ns', 'TTFT_Std_ns',
            'TBT_Mean_ns', 'TBT_Median_ns', 'TBT_Min_ns', 'TBT_Max_ns', 'TBT_Std_ns',
            'Latency_Mean_ns', 'Latency_Median_ns', 'Latency_Min_ns', 'Latency_Max_ns', 'Latency_Std_ns',
            'TTFT_Mean_Formatted', 'TBT_Mean_Formatted', 'Latency_Mean_Formatted', 'Max_Completion_Time_Formatted'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for group_name, metrics in results.items():
            row = {
                'Group': group_name,
                'Request_Count': metrics['request_count'],
                'Token_Count_Per_Request': metrics['token_count_per_request'],
                'Total_Tokens': metrics['total_tokens'],
                'Max_Completion_Time_ns': metrics['max_completion_time_ns'],
                'Throughput_Tokens_Per_Second': f"{metrics['throughput_tokens_per_second']:.3f}",
                'TTFT_Mean_ns': metrics['ttft']['mean'],
                'TTFT_Median_ns': metrics['ttft']['median'],
                'TTFT_Min_ns': metrics['ttft']['min'],
                'TTFT_Max_ns': metrics['ttft']['max'],
                'TTFT_Std_ns': metrics['ttft']['std'],
                'TBT_Mean_ns': metrics['tbt']['mean'],
                'TBT_Median_ns': metrics['tbt']['median'],
                'TBT_Min_ns': metrics['tbt']['min'],
                'TBT_Max_ns': metrics['tbt']['max'],
                'TBT_Std_ns': metrics['tbt']['std'],
                'Latency_Mean_ns': metrics['latency']['mean'],
                'Latency_Median_ns': metrics['latency']['median'],
                'Latency_Min_ns': metrics['latency']['min'],
                'Latency_Max_ns': metrics['latency']['max'],
                'Latency_Std_ns': metrics['latency']['std'],
                'TTFT_Mean_Formatted': format_time(metrics['ttft']['mean']),
                'TBT_Mean_Formatted': format_time(metrics['tbt']['mean']),
                'Latency_Mean_Formatted': format_time(metrics['latency']['mean']),
                'Max_Completion_Time_Formatted': format_time(metrics['max_completion_time_ns'])
            }
            writer.writerow(row)

def save_results_to_json(results, output_path):
    """
    将结果保存为JSON文件
    """
    # 为每个组添加格式化的时间
    formatted_results = {}
    for group_name, metrics in results.items():
        formatted_metrics = metrics.copy()
        for metric_type in ['ttft', 'tbt', 'latency']:
            formatted_metrics[f'{metric_type}_formatted'] = {
                key: format_time(value) for key, value in metrics[metric_type].items()
            }
        # 添加throughput相关的格式化信息
        formatted_metrics['max_completion_time_formatted'] = format_time(metrics['max_completion_time_ns'])
        formatted_metrics['throughput_formatted'] = f"{metrics['throughput_tokens_per_second']:.3f} tokens/s"
        formatted_results[group_name] = formatted_metrics
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, indent=2, ensure_ascii=False)

def print_summary(results):
    """
    打印分析结果摘要
    """
    print("\n" + "="*80)
    print("数据分析结果摘要")
    print("="*80)
    
    for group_name, metrics in results.items():
        print(f"\n📊 组: {group_name}")
        print(f"   请求数量: {metrics['request_count']}")
        print(f"   每请求token数: {metrics['token_count_per_request']}")
        print(f"   总token数: {metrics['total_tokens']}")
        
        print(f"\n   🚀 Throughput:")
        print(f"      吞吐量: {metrics['throughput_tokens_per_second']:.3f} tokens/秒")
        print(f"      最晚完成时间: {format_time(metrics['max_completion_time_ns'])}")
        
        print(f"\n   ⏱️  TTFT (Time to First Token):")
        print(f"      平均值: {format_time(metrics['ttft']['mean'])}")
        print(f"      中位数: {format_time(metrics['ttft']['median'])}")
        print(f"      标准差: {format_time(metrics['ttft']['std'])}")
        
        print(f"\n   🔄 TBT (Time Between Tokens):")
        print(f"      平均值: {format_time(metrics['tbt']['mean'])}")
        print(f"      中位数: {format_time(metrics['tbt']['median'])}")
        print(f"      标准差: {format_time(metrics['tbt']['std'])}")
        
        print(f"\n   📈 Latency (Total Time):")
        print(f"      平均值: {format_time(metrics['latency']['mean'])}")
        print(f"      中位数: {format_time(metrics['latency']['median'])}")
        print(f"      标准差: {format_time(metrics['latency']['std'])}")
        
        print("-" * 60)

def main():
    # 输入文件路径
    input_file = input("请输入数据文件路径 (默认: data.txt): ").strip()
    if not input_file:
        input_file = "data.txt"
    input_file = "pdhete/" + input_file + ".txt"
    
    # 检查文件是否存在
    if not Path(input_file).exists():
        print(f"错误: 文件 '{input_file}' 不存在!")
        return
    
    print(f"正在分析文件: {input_file}")
    
    # 解析数据
    try:
        parsed_data = parse_data_file(input_file)
        print(f"成功解析 {len(parsed_data)} 个数据组")
    except Exception as e:
        print(f"解析文件时出错: {e}")
        return
    
    # 计算指标
    results = {}
    for group_name, group_data in parsed_data.items():
        print(f"正在处理组: {group_name}")
        metrics = calculate_metrics(group_data)
        if metrics:
            results[group_name] = metrics
    
    if not results:
        print("没有找到有效的数据进行分析!")
        return
    
    # 打印摘要
    print_summary(results)
    
    # 保存结果
    output_base = Path(input_file).stem
    
    # 保存为CSV
    # csv_output = f"{output_base}_analysis_results.csv"
    # save_results_to_csv(results, csv_output)
    # print(f"\n✅ CSV结果已保存到: {csv_output}")
    
    # 保存为JSON
    json_output = f"pdhete/{output_base}_analysis_results.json"
    save_results_to_json(results, json_output)
    print(f"✅ JSON结果已保存到: {json_output}")
    
    print(f"\n🎉 分析完成! 共处理了 {len(results)} 个数据组")

if __name__ == "__main__":
    main()