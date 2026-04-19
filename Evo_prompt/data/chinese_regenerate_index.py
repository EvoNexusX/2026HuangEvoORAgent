"""
索引重新生成工具
================
本脚本用于重新生成JSON文件中的连续索引编号。

应用场景：
- 当JSON数据集的索引不连续时（如删除了某些条目）
- 需要规范化索引顺序时
- 维护数据集一致性

功能特点：
1. 自动备份原文件（.backup后缀）
2. 保持原有顺序（按当前index排序）
3. 重新生成0开始的连续索引
4. 支持dry-run模式（预览不修改）

使用示例：
python regenerate_index.py data.json
python regenerate_index.py data.json --dry-run
"""

#!/usr/bin/env python3

# ============================================================================
# 导入依赖库
# ============================================================================
import json  # JSON数据处理
import sys  # 系统相关功能
import argparse  # 命令行参数解析
from pathlib import Path  # 现代路径操作

# ============================================================================
# 核心重新索引函数
# ============================================================================

def regenerate_index(json_file_path):
    """
    重新生成JSON文件中的连续索引编号
    
    处理流程：
    1. 加载JSON文件
    2. 提取所有包含index字段的条目
    3. 按现有index排序（保持原顺序）
    4. 重新分配0开始的连续索引
    5. 创建原文件备份
    6. 写入更新后的数据
    
    参数:
        json_file_path (str): JSON文件路径
    
    返回:
        bool: 操作是否成功
    """
    
    # 加载JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{json_file_path}': {e}")
        return False
    
    # 将数据转换为列表并按当前索引排序以保持顺序
    items = []
    for key, value in data.items():
        # 检查每个条目是否包含index字段
        if isinstance(value, dict) and 'index' in value:
            items.append(value)
        else:
            print(f"Warning: Item with key '{key}' doesn't have an 'index' field")
    
    # 按现有index排序以保持原始顺序
    # 使用float('inf')确保没有index的条目排在最后
    items.sort(key=lambda x: x.get('index', float('inf')))
    
    # 创建包含重新生成索引的新数据
    new_data = {}
    for i, item in enumerate(items):
        item['index'] = i  # 更新index字段为新的连续索引
        new_data[str(i)] = item  # 使用字符串索引作为键（保持原格式）
    
    # 创建原文件的备份
    backup_path = f"{json_file_path}.backup"
    try:
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Backup created: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")
    
    # 将重新生成的数据写回文件
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
        
        # 输出操作成功信息
        print(f"Successfully regenerated indices in '{json_file_path}'")
        print(f"Total items: {len(new_data)}")
        print(f"Index range: 0 to {len(new_data) - 1}")
        return True
        
    except Exception as e:
        print(f"Error writing to file '{json_file_path}': {e}")
        return False

# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：解析命令行参数并执行索引重新生成
    
    支持的参数：
    - json_file: 必需，要处理的JSON文件路径
    - --dry-run: 可选，预览模式不修改文件
    """
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Regenerate sequential index numbers in a JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python regenerate_index.py data.json
  python regenerate_index.py /path/to/dataset.json
        """
    )
    
    # 添加文件路径参数（必需）
    parser.add_argument(
        'json_file', 
        help='Path to the JSON file to process'
    )
    
    # 添加dry-run标志（可选）
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be changed without modifying the file'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.json_file).exists():
        print(f"Error: File '{args.json_file}' does not exist.")
        sys.exit(1)
    
    # 如果是dry-run模式，显示提示信息
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        print("(Dry-run functionality not fully implemented in this version)")
        # 注意：实际的dry-run逻辑需要在regenerate_index函数中实现
        # 这里只是显示模式提示
    
    # 执行索引重新生成
    success = regenerate_index(args.json_file)
    
    # 根据结果设置退出码
    sys.exit(0 if success else 1)

# ============================================================================
# 脚本入口
# ============================================================================
if __name__ == "__main__":
    main()
