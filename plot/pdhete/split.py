import re
import os

def split_experiment_data(input_file_path, output_dir="experiment_data"):
    """
    将包含多组实验数据的txt文件分割成单独的文件
    
    Args:
        input_file_path (str): 输入文件路径
        output_dir (str): 输出目录路径，默认为"experiment_data"
    """
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    try:
        # 读取输入文件
        with open(input_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 使用正则表达式找到所有实验名和对应数据
        pattern = r'\*([^*]+)\*'
        
        # 找到所有匹配的实验名位置
        matches = list(re.finditer(pattern, content))
        
        if not matches:
            print("未找到符合 *实验名* 格式的标题")
            return
        
        experiment_count = 0
        
        for i, match in enumerate(matches):
            experiment_name = match.group(1).strip()
            
            # 确定当前实验数据的起始位置（实验名之后）
            data_start = match.end()
            
            # 确定当前实验数据的结束位置（下一个实验名之前，或文件结尾）
            if i + 1 < len(matches):
                data_end = matches[i + 1].start()
            else:
                data_end = len(content)
            
            # 提取实验数据
            experiment_data = content[data_start:data_end].strip()
            
            # 如果数据为空，跳过
            if not experiment_data:
                print(f"警告: 实验 '{experiment_name}' 没有数据内容")
                continue
            
            # 清理实验名，移除不适合用作文件名的字符
            clean_name = re.sub(r'[<>:"/\\|?*]', '_', experiment_name)
            clean_name = clean_name.replace(' ', '_')
            
            # 生成输出文件名，如果文件已存在则添加序号
            base_filename = f"{clean_name}.txt"
            output_filename = base_filename
            counter = 1
            
            while os.path.exists(os.path.join(output_dir, output_filename)):
                name_without_ext = clean_name
                output_filename = f"{name_without_ext}_{counter}.txt"
                counter += 1
            
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"处理实验: '{experiment_name}' -> 文件: {output_filename}")
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as output_file:
                # 可以选择是否在文件开头包含实验名
                output_file.write(f"实验名: {experiment_name}\n")
                output_file.write("=" * 50 + "\n")
                output_file.write(experiment_data)
            
            experiment_count += 1
            print(f"已创建文件: {output_path}")
        
        print(f"\n处理完成！共分割出 {experiment_count} 个实验数据文件")
        
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{input_file_path}'")
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")

def main():
    # 使用示例
    print("实验数据分割工具")
    print("-" * 30)
    
    # 获取输入文件路径
    input_file = input("请输入txt文件路径: ").strip()
    
    # 如果用户直接按回车，使用默认文件名
    if not input_file:
        input_file = "experiment_data.txt"
        print(f"使用默认文件名: {input_file}")
    
    # 获取输出目录（可选）
    output_dir = input("请输入输出目录名（直接回车使用默认'experiment_data'）: ").strip()
    if not output_dir:
        output_dir = "experiment_data"
    
    # 执行分割
    split_experiment_data(input_file, output_dir)

if __name__ == "__main__":
    main()

# 如果你想直接在代码中指定文件路径，可以使用以下方式：
# split_experiment_data("your_file.txt", "output_folder")