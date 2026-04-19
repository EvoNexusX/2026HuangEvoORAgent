"""
数据集格式转换工具：IndustryOR格式转Default格式
===============================================
本脚本用于将executed.jsonl（IndustryOR格式）转换为dataset_combined_result.json格式（Default格式）。

转换规则：
- "en_question" → "question"
- "en_answer" → "answer" 
- 生成连续的"index"字段（从0开始）
- 输出格式：标准JSON（带缩进，便于阅读）

主要用于将IndustryOR数据集转换为统一的评估格式。
"""

#!/usr/bin/env python3

# ============================================================================
# 导入依赖库
# ============================================================================
import json  # JSON数据处理
import os  # 操作系统相关功能，用于路径处理

# ============================================================================
# 核心转换函数
# ============================================================================

def convert_executed_to_dataset_format(input_file, output_file):
    """
    将executed.jsonl格式转换为dataset_combined_result.json格式
    
    读取JSONL格式的IndustryOR数据（每行一个JSON对象），
    提取英文问题和答案，生成连续索引，
    输出为标准JSON格式（字典结构）。
    
    参数:
        input_file: 输入文件路径（executed.jsonl）
        output_file: 输出文件路径（JSON格式）
    
    返回:
        bool: 转换是否成功
    """
    
    converted_data = {}  # 存储转换后的数据
    index = 0  # 连续索引，从0开始
    
    try:
        # 逐行读取JSONL文件
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()  # 去除首尾空白
                if not line:  # 跳过空行
                    continue
                
                try:
                    # 解析JSON行
                    data = json.loads(line)
                    
                    # 转换格式
                    converted_entry = {
                        "index": index,  # 连续索引
                        "question": data.get("en_question", ""),  # 提取英文问题
                        "answer": data.get("en_answer", "")  # 提取英文答案
                    }
                    
                    # 添加到转换数据字典（使用字符串索引作为键）
                    converted_data[str(index)] = converted_entry
                    index += 1
                    
                    # 每处理100条打印进度
                    if index % 100 == 0:
                        print(f"Processed {index} entries...")
                        
                except json.JSONDecodeError as e:
                    # JSON解析错误
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    print(f"Problematic line: {line[:100]}...")  # 只显示前100个字符
                    continue
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error reading input file: {e}")
        return False
    
    # 将转换后的数据写入输出文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # 使用indent=4进行格式化输出（便于阅读）
            json.dump(converted_data, f, ensure_ascii=False, indent=4)
        
        print(f"Conversion completed successfully!")
        print(f"Total entries processed: {index}")
        print(f"Output saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False

# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：定义文件路径并执行转换
    
    包含多个数据集的转换配置（注释掉的部分），
    可以根据需要取消注释来转换不同的数据集。
    """
    
    # 定义文件路径配置（可以切换不同数据集）
    # IndustryOR数据集
    # input_file = "data/datasets/ORLM/IndustryOR.q2mc_en.ORLM-LLaMA-3-8B/executed.jsonl"
    # output_file = "data/datasets/ORLM/IndustryOR.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
    
    # MAMO.ComplexLP数据集
    # input_file = "data/datasets/ORLM/MAMO.ComplexLP.q2mc_en.ORLM-LLaMA-3-8B/executed.jsonl"
    # output_file = "data/datasets/ORLM/MAMO.ComplexLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
    
    # MAMO.EasyLP数据集
    # input_file = "data/datasets/ORLM/MAMO.EasyLP.q2mc_en.ORLM-LLaMA-3-8B/executed.jsonl"
    # output_file = "data/datasets/ORLM/MAMO.EasyLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
    
    # NL4OPT数据集（当前激活）
    input_file = "data/datasets/ORLM/NL4OPT.q2mc_en.ORLM-LLaMA-3-8B/executed.jsonl"
    output_file = "data/datasets/ORLM/NL4OPT.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return
    
    print(f"Converting {input_file} to {output_file}...")
    print("=" * 50)
    
    # 执行转换
    success = convert_executed_to_dataset_format(input_file, output_file)
    
    if success:
        print("=" * 50)
        print("Conversion completed successfully!")
        
        # 显示转换后数据的示例
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                converted_data = json.load(f)
                
            print(f"\nSample of converted data (first entry):")
            if "0" in converted_data:
                sample_entry = converted_data["0"]
                print(f"Index: {sample_entry['index']}")
                print(f"Question (first 100 chars): {sample_entry['question'][:100]}...")
                print(f"Answer: {sample_entry['answer']}")
            
        except Exception as e:
            print(f"Error reading converted file for sample: {e}")
    else:
        print("Conversion failed. Please check the error messages above.")

# ============================================================================
# 脚本入口
# ============================================================================
if __name__ == "__main__":
    main()
