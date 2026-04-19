"""
数据集格式转换工具：Default格式转IndustryOR格式
=======================================================
本脚本用于将Default.json和Default-en.json两个文件合并转换为IndustryOR.json格式。

转换规则：
- "id": 来自"index"字段
- "en_question": 来自Default-en.json的"question"字段
- "cn_question": 来自Default.json的"question"字段  
- "en_answer": 来自"answer"字段（两个文件相同）
- "difficulty": 统一设置为"Medium"

输出格式：JSONL（每行一个JSON对象）
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

def convert_default_to_industry_format(cn_file, en_file, output_file):
    """
    将Default格式转换为IndustryOR格式
    
    读取中文和英文两个JSON文件，合并信息并转换为IndustryOR的JSONL格式。
    IndustryOR格式包含双语问题和答案，以及难度等级。
    
    参数:
        cn_file: Default.json文件路径（包含中文问题）
        en_file: Default-en.json文件路径（包含英文问题）
        output_file: 输出文件路径
    """
    
    # 读取中文版本（Default.json）
    with open(cn_file, 'r', encoding='utf-8') as f:
        cn_data = json.load(f)
    
    # 读取英文版本（Default-en.json）
    with open(en_file, 'r', encoding='utf-8') as f:
        en_data = json.load(f)
    
    # 存储转换后的数据
    converted_data = []
    
    # 遍历所有条目（假设两个文件的键相同）
    for key in cn_data.keys():
        if key in en_data:
            cn_entry = cn_data[key]  # 中文数据
            en_entry = en_data[key]  # 英文数据
            
            # 创建IndustryOR格式的新条目
            new_entry = {
                "en_question": en_entry["question"],  # 英文问题
                "cn_question": cn_entry["question"],  # 中文问题
                "en_answer": str(en_entry["answer"]),  # 英文答案（转为字符串）
                "difficulty": "Medium",  # 难度等级（统一设为Medium）
                "id": cn_entry["index"]  # ID（来自index字段）
            }
            
            converted_data.append(new_entry)
        else:
            # 警告：英文数据中找不到对应的键
            print(f"Warning: Key {key} not found in English data")
    
    # 按ID排序，保持顺序
    converted_data.sort(key=lambda x: x["id"])
    
    # 以JSONL格式写入输出文件（每行一个JSON对象）
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in converted_data:
            # 使用separators参数去除多余空格，使JSON更紧凑
            json.dump(entry, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')  # 每个对象后换行
    
    print(f"Successfully converted {len(converted_data)} entries.")
    print(f"Output written to: {output_file}")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：执行转换流程
    
    定义输入输出文件路径，检查文件存在性，然后执行转换。
    """
    
    # 定义文件路径（相对于脚本位置）
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
    cn_file = os.path.join(script_dir, "datasets", "Default.json")  # 中文文件
    en_file = os.path.join(script_dir, "datasets", "Default-en.json")  # 英文文件
    output_file = os.path.join(script_dir, "datasets", "Default_converted.json")  # 输出文件
    
    # 检查输入文件是否存在
    if not os.path.exists(cn_file):
        print(f"Error: Chinese file not found: {cn_file}")
        return
        
    if not os.path.exists(en_file):
        print(f"Error: English file not found: {en_file}")
        return
    
    # 执行转换
    try:
        convert_default_to_industry_format(cn_file, en_file, output_file)
    except Exception as e:
        print(f"Error during conversion: {e}")

# ============================================================================
# 脚本入口
# ============================================================================
if __name__ == "__main__":
    main()
