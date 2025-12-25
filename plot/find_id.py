import re

def find_unmatched_start_ids(log_file_path):
    start_count = {}
    end_count = {}

    # 正则表达式匹配 start 和 end 行，并提取 id
    pattern = re.compile(r'(start|end).*id\s+(\d+)')

    with open(log_file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            match = pattern.search(line)
            if match:
                keyword, id_str = match.groups()
                log_id = int(id_str)
                if keyword == 'start':
                    start_count[log_id] = start_count.get(log_id, 0) + 1
                elif keyword == 'end':
                    end_count[log_id] = end_count.get(log_id, 0) + 1

    # 找出 start 比 end 多的 id
    unmatched_ids = {}
    all_ids = set(start_count.keys()) | set(end_count.keys())
    for log_id in all_ids:
        start_num = start_count.get(log_id, 0)
        end_num = end_count.get(log_id, 0)
        if start_num > end_num:
            unmatched_ids[log_id] = start_num - end_num

    return unmatched_ids

# 使用脚本
if __name__ == '__main__':
    log_file = 'log.txt'
    result = find_unmatched_start_ids(log_file)
    if result:
        print("以下 id 的 start 比 end 多：")
        for log_id, count in result.items():
            print(f"ID {log_id}: 多出 {count} 个 start")
    else:
        print("所有 start 都有对应的 end，没有多出的 start。")