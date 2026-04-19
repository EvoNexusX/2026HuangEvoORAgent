"""
通用数据集处理工具
==================
本脚本用于处理包含多个问题目录的数据集，每个问题目录包含：
- description.txt: 问题描述
- sample.json: 样本输入输出

主要功能：
1. 遍历所有问题目录（支持编号目录prob_*或命名目录）
2. 读取问题描述和样本数据
3. 合并input到问题描述中
4. 输出统一的JSON格式文件

使用示例：
python process_dataset.py data/datasets/LPWP output.json
python process_dataset.py data/datasets/ComplexOR output.json --named
"""

#!/usr/bin/env python3

# ============================================================================
# 导入依赖库
# ============================================================================
import os  # 操作系统相关功能，路径处理
import json  # JSON数据处理
import glob  # 文件路径模式匹配
import argparse  # 命令行参数解析

# ============================================================================
# 核心处理函数
# ============================================================================

def process_dataset(dataset_path, output_file, is_numbered=True):
    """
    处理包含问题目录的数据集，合并为单一JSON文件
    
    每个问题目录包含：
    - description.txt: 问题描述文本
    - sample.json: 样本数据（包含input和output字段）
    
    处理流程：
    1. 根据is_numbered参数查找目录（prob_*或所有子目录）
    2. 排序目录（按数字或字母）
    3. 读取每个目录中的description.txt和sample.json
    4. 将input数据格式化并附加到问题描述中
    5. 提取output作为答案
    6. 输出为统一的JSON格式
    
    参数:
        dataset_path (str): 数据集根目录路径
        output_file (str): 输出JSON文件路径
        is_numbered (bool): 目录是否使用数字索引（如prob_10），False表示命名目录
    """
    
    # 获取所有问题目录
    if is_numbered:
        # 查找所有prob_*格式的目录
        prob_dirs = glob.glob(os.path.join(dataset_path, "prob_*"))
        # 按问题编号排序（提取目录名最后的数字）
        prob_dirs.sort(key=lambda x: int(x.split("_")[-1]))
    else:
        # 查找所有子目录（命名目录）
        prob_dirs = glob.glob(os.path.join(dataset_path, "*"))
        # 过滤出目录（排除文件）
        prob_dirs = [d for d in prob_dirs if os.path.isdir(d)]
        # 按字母顺序排序
        prob_dirs.sort()
    
    # 存储所有处理后的问题数据
    combined_data = {}
    
    # 遍历处理每个问题目录
    for i, prob_dir in enumerate(prob_dirs):
        # 提取问题标识符（目录名或编号）
        prob_id = os.path.basename(prob_dir)
        
        # 构建描述文件和样本文件的路径
        description_path = os.path.join(prob_dir, "description.txt")
        sample_path = os.path.join(prob_dir, "sample.json")
        
        # 检查必需文件是否存在
        if not (os.path.exists(description_path) and os.path.exists(sample_path)):
            print(f"Warning: Missing files in {prob_dir}. Skipping.")
            continue
        
        # 读取问题描述（从description.txt）
        with open(description_path, 'r', encoding='utf-8') as f:
            question = f.read().strip()
        
        # 读取样本数据中的答案和输入（从sample.json）
        try:
            with open(sample_path, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
                
                # 检查sample.json格式：应该是包含字典的列表
                if isinstance(sample_data, list) and len(sample_data) > 0:
                    # 提取output字段作为答案
                    if 'output' in sample_data[0]:
                        answer = sample_data[0]['output']
                        # 如果答案是只有一个元素的列表，提取该元素
                        if isinstance(answer, list) and len(answer) == 1:
                            answer = answer[0]
                    else:
                        print(f"Warning: No 'output' field in {prob_dir}/sample.json. Skipping.")
                        continue
                    
                    # 如果有input数据，将其格式化后附加到问题描述中
                    if 'input' in sample_data[0]:
                        input_data = sample_data[0]['input']
                        # 将input数据转换为格式化的JSON字符串
                        input_str = json.dumps(input_data, indent=2)
                        question += f"\n\nInput:\n{input_str}"
                else:
                    print(f"Warning: Unexpected sample.json format in {prob_dir}. Skipping.")
                    continue
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {prob_dir}/sample.json. Skipping.")
            continue
        
        # 将处理后的数据添加到combined_data中
        combined_data[str(i)] = {
            "index": i,  # 问题索引（从0开始）
            "question": question,  # 问题描述（包含input数据）
            "answer": answer  # 期望答案
        }
    
    # 如果输出目录不存在，创建它
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 将合并后的数据写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)
    
    # 输出处理结果统计
    print(f"Processing complete. Output written to {output_file}")
    print(f"Processed {len(combined_data)} problems out of {len(prob_dirs)} directories")

# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='Process a dataset of problems into a combined JSON file.')
    
    # 必需参数
    parser.add_argument('dataset_path', help='Path to the dataset directory')
    parser.add_argument('output_file', help='Path to the output JSON file')
    
    # 可选参数：--named标志用于处理命名目录（而非编号目录）
    parser.add_argument('--named', action='store_true', 
                        help='Use if directories have names instead of numeric indices')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 执行数据集处理
    # 注意：not args.named 表示如果指定了--named，则is_numbered=False
    process_dataset(args.dataset_path, args.output_file, not args.named)
