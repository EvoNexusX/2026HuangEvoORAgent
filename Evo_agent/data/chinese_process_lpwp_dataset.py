"""
LPWP数据集处理工具
==================
本脚本专门用于处理LPWP (Linear Programming Word Problems)数据集。

数据集结构：
- data/datasets/LPWP/
  - prob_1/
    - description.txt
    - sample.json
  - prob_2/
    - description.txt
    - sample.json
  - ...

主要功能：
1. 遍历LPWP目录下的所有prob_*格式的子目录
2. 按问题编号排序
3. 读取每个问题的描述和样本数据
4. 合并input到问题描述，提取output作为答案
5. 输出为lpwp_combined_result.json

输出格式：
{
  "0": {"index": 0, "question": "...", "answer": "..."},
  "1": {"index": 1, "question": "...", "answer": "..."},
  ...
}
"""

#!/usr/bin/env python3

# ============================================================================
# 导入依赖库
# ============================================================================
import os  # 操作系统相关功能
import json  # JSON数据处理
import glob  # 文件路径模式匹配

# ============================================================================
# 核心处理函数
# ============================================================================

def process_lpwp_dataset():
    """
    处理LPWP数据集，合并为单一JSON文件
    
    LPWP数据集包含多个prob_*编号目录，每个目录包含：
    - description.txt: 问题描述
    - sample.json: 样本数据（包含input和output）
    
    处理流程：
    1. 查找data/datasets/LPWP下的所有prob_*目录
    2. 按问题编号排序
    3. 读取每个目录的description.txt和sample.json
    4. 将input数据附加到问题描述中
    5. 提取output作为答案
    6. 输出到lpwp_combined_result.json
    """
    
    # LPWP数据集路径
    lpwp_path = "data/datasets/LPWP"
    
    # 获取所有prob_*格式的目录
    prob_dirs = glob.glob(os.path.join(lpwp_path, "prob_*"))
    
    # 按问题编号排序（提取目录名最后的数字）
    # 例如：prob_1, prob_2, prob_10 按数字大小排序
    prob_dirs.sort(key=lambda x: int(x.split("_")[-1]))
    
    # 存储所有处理后的问题数据
    combined_data = {}
    
    # 遍历处理每个问题目录
    for i, prob_dir in enumerate(prob_dirs):
        # 提取问题编号（目录名最后的数字）
        prob_number = prob_dir.split("_")[-1]
        
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
    
    # 输出文件路径
    output_path = "data/datasets/lpwp_combined_result.json"
    
    # 将合并后的数据写入JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)
    
    # 输出处理结果统计
    print(f"Processing complete. Output written to {output_path}")
    print(f"Processed {len(combined_data)} problems out of {len(prob_dirs)} directories")

# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    process_lpwp_dataset()
