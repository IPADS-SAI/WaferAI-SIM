import os

# 目标文件列表
target_files = ['static/index.html', 'static/timeline.js', 'main.py']

# 输出文件名
output_file = 'llm_input.txt'

def get_directory_tree(start_path='.', indent=''):
    """递归生成目录树结构"""
    tree_lines = []
    items = sorted(os.listdir(start_path))
    for item in items:
        path = os.path.join(start_path, item)
        if os.path.isdir(path):
            tree_lines.append(f"{indent}├── {item}/")
            tree_lines.extend(get_directory_tree(path, indent + "│   "))
        else:
            tree_lines.append(f"{indent}├── {item}")
    return tree_lines

def write_llm_input():
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # 写入目录结构
        out_f.write("=== 目录结构 ===\n")
        tree_lines = get_directory_tree()
        for line in tree_lines:
            out_f.write(line + '\n')
        out_f.write("\n" + "="*50 + "\n\n")

        # 写入每个目标文件内容
        for filename in target_files:
            if os.path.exists(filename):
                out_f.write(f"=== {filename} ===\n")
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    out_f.write(content)
                out_f.write("\n" + "="*50 + "\n\n")
            else:
                out_f.write(f"=== {filename} ===\n")
                out_f.write(f"[文件不存在]\n\n" + "="*50 + "\n\n")

    print(f"✅ 已生成 {output_file}")

if __name__ == "__main__":
    write_llm_input()