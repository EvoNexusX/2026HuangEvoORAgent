"""
运筹优化问题类型分析和可视化工具
=====================================
本脚本用于分析JSON数据集中运筹优化问题的类型（Type）和问题分类（Problem）分布情况，
并生成多种可视化图表（水平条形图、饼图等）。

主要功能：
1. 从JSON文件中读取并统计问题类型和分类
2. 生成带标签的水平条形图展示类型分布
3. 生成饼图（带

和不带标签）展示问题分类分布
4. 使用专业的配色和排版，适合学术论文和报告
5. 支持中文等宽字符的正确显示
"""

# ============================================================================
# 导入依赖库
# ============================================================================
import json  # JSON数据处理
from collections import Counter  # 计数器，用于统计数据出现频率
import matplotlib.pyplot as plt  # 绘图库
import matplotlib.colors as mcolors  # 颜色处理
import matplotlib.font_manager as fm  # 字体管理
import numpy as np  # 数值计算库

# ============================================================================
# 数据分析函数
# ============================================================================

def analyze_json_data(file_path):
    """
    分析JSON文件中的类型（Type）和问题分类（Problem）分布
    
    从JSON数据集中提取每个条目的'type'和'problem'字段，
    统计它们的出现次数，用于后续的可视化和分析。
    
    参数:
        file_path: JSON数据文件的路径
    
    返回:
        (types_counter, problems_counter): 两个Counter对象
            - types_counter: 类型统计（如：线性规划、整数规划等）
            - problems_counter: 问题统计（如：资源分配、调度问题等）
    """
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 初始化计数器
    types_counter = Counter()  # 统计问题类型
    problems_counter = Counter()  # 统计问题分类
    
    # 遍历数据集中的每个条目，统计类型和问题出现次数
    for key, item in data.items():
        if isinstance(item, dict) and 'type' in item and 'problem' in item:
            types_counter[item['type']] += 1
            problems_counter[item['problem']] += 1
    
    return types_counter, problems_counter

def print_counts(counter, label):
    """
    打印统计信息到控制台
    
    以格式化的方式显示各类别的数量、唯一类别数和总数。
    
    参数:
        counter: Counter对象，包含统计数据
        label: 标签名称（如"Type"或"Problem"）
    """
    print(f"\n{label} Distribution:")
    print("-" * 50)
    # 按出现次数从高到低排序
    for item, count in counter.most_common():
        print(f"{item}: {count}")
    print(f"Total unique {label.lower()}: {len(counter)}")
    print(f"Total count: {sum(counter.values())}")

# ============================================================================
# 可视化函数
# ============================================================================

def plot_horizontal_bar(counter, title, filename):
    """
    绘制水平条形图展示类型分布百分比
    
    生成专业的水平条形图，包含：
    - 蓝绿色（Teal）配色方案
    - 百分比标签
    - Times New Roman字体
    - 网格线辅助阅读
    - 适合学术论文的高质量输出
    
    参数:
        counter: Counter对象，包含要可视化的数据
        title: 图表标题
        filename: 保存文件的路径
    """
    # 获取按计数排序的数据（从高到低）
    items_sorted = counter.most_common()
    
    # 反转顺序，使得最大值出现在顶部
    items_sorted.reverse()
    
    # 解包为标签和数值
    labels, values = zip(*items_sorted)
    
    # 计算每个项目的百分比
    total = sum(values)
    percentages = [(v/total)*100 for v in values]
    
    # 设置全局字体为Times New Roman（学术论文标准）
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14  # 基础字体大小
    
    # 使用seaborn白网格风格，提供清晰的背景
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))  # 增加图表尺寸以容纳更多信息
    
    # 使用蓝绿色（Teal）作为主色调
    teal_colors = '#2E8B8B'
    
    # 创建水平条形图
    bars = ax.barh(labels, percentages, color=teal_colors, height=0.7, edgecolor='none')
    
    # 移除所有边框线，使图表更简洁
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 在每个条形末端添加百分比标签
    for i, (bar, percentage) in enumerate(zip(bars, percentages)):
        ax.text(
            percentage + 0.5,  # 条形末端稍微偏右
            bar.get_y() + bar.get_height()/2,  # 垂直居中
            f"{percentage:.1f}%",  # 格式：保留一位小数
            va='center',  # 垂直对齐：居中
            ha='left',  # 水平对齐：左对齐
            fontsize=28,  # 较大的字体以便阅读
            color='#444444',  # 深灰色
            family='Times New Roman',
            fontweight='normal'
        )
    
    # 设置标题（左对齐，使用Teal色系）
    ax.set_title(title, fontsize=26, color='#2a7f90', pad=20, loc='left', family='Times New Roman')
    
    # 设置x轴标签
    ax.set_xlabel('Percentage of Types', fontsize=28, color='#444444', family='Times New Roman')
    
    # 设置y轴刻度字体大小（类别标签）
    ax.tick_params(axis='y', which='both', left=False, labelsize=30)
    ax.tick_params(axis='x', labelsize=22)
    
    # 显式设置y轴刻度标签字体为Times New Roman
    for tick in ax.get_yticklabels():
        tick.set_fontname('Times New Roman')
    
    # 显式设置x轴刻度标签字体为Times New Roman
    for tick in ax.get_xticklabels():
        tick.set_fontname('Times New Roman')
    
    # 设置x轴网格线（浅灰色，仅显示主刻度）
    ax.grid(axis='x', color='#EEEEEE', linestyle='-', linewidth=0.7, alpha=0.7)
    ax.grid(axis='y', visible=False)  # 不显示y轴网格线
    
    # 设置x轴范围（留出空间显示标签）
    ax.set_xlim(0, max(percentages) * 1.15)
    
    # 调整布局，避免标签被裁剪
    plt.tight_layout()
    
    # 保存图表（高分辨率，白色背景）
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Chart saved as '{filename}'")
    
    # 关闭图表释放内存
    plt.close(fig)

def plot_pie_chart(counter, title, filename):
    """
    绘制带标签的环形饼图
    
    生成专业的环形饼图（Donut Chart），特点：
    - 环形设计，中心显示总数
    - 不同类别使用专门配色（线性规划-Teal，整数规划-深蓝，混合整数规划-深红）
    - 带有连接线的外部标签和百分比
    - 适合展示问题类别的分布情况
    
    参数:
        counter: Counter对象，包含要可视化的数据
        title: 图表标题
        filename: 保存文件的路径
    """
    # 获取按计数排序的数据
    labels, values = zip(*counter.most_common())
    
    # 计算每个项目的百分比
    total = sum(values)
    percentages = [(v/total)*100 for v in values]
    
    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 创建图表
    plt.figure(figsize=(10, 10), facecolor='white')
    ax = plt.subplot(111)
    
    # 为不同问题类型定义专门的颜色
    color_mapping = {
        'Linear programming': '#2E8B8B',      # 蓝绿色
        'Integer programming': '#2E318B',     # 深蓝色
        'Mixed integer programming': '#8B1A1A'  # 深红色
    }
    
    # 将颜色映射到实际标签
    colors = [color_mapping.get(label, '#999999') for label in labels]
    
    # 创建饼图，只捕获wedges（扇形）
    wedges = ax.pie(
        values, 
        colors=colors,
        startangle=90,  # 从顶部开始
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True},
        labels=None,  # 不在饼图上直接显示标签
        autopct=None  # 不在饼图上直接显示百分比
    )[0]
    
    # 在中心绘制白色圆形，创建环形效果
    centre_circle = plt.Circle((0, 0), 0.5, fc='white', edgecolor='none')
    ax.add_patch(centre_circle)
    
    # 在中心添加总数标签
    ax.text(0, 0, f"n = {total}", ha='center', va='center', fontsize=20, family='Times New Roman')
    
    # 为每个扇形添加标签和百分比（使用连接线）
    for i, (wedge, label, pct) in enumerate(zip(wedges, labels, percentages)):
        # 获取角度和半径
        ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        
        # 根据角度确定水平对齐方式
        ha = 'left' if x >= 0 else 'right'
        
        # 连接点在扇形边缘
        conn_x = 0.75 * x
        conn_y = 0.75 * y
        
        # 文本位置在外部
        text_x = 1.2 * x
        text_y = 1.2 * y
        
        # 绘制连接线
        ax.plot([conn_x, text_x], [conn_y, text_y], color='gray', linewidth=0.8)
        
        # 添加标签
        ax.text(text_x, text_y, label, ha=ha, va='center', fontsize=15, color='#444444', family='Times New Roman')
        
        # 在标签下方添加百分比
        ax.text(text_x, text_y - 0.15, f"{pct:.1f}%", ha=ha, va='center', fontsize=18, color='#666666', family='Times New Roman')
    
    # 设置标题
    ax.set_title(title, fontsize=22, color='#444444', pad=20, loc='center', y=1.05, family='Times New Roman')
    
    # 确保饼图为圆形
    ax.axis('equal')
    
    # 移除所有边框和刻度
    ax.set_frame_on(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved as '{filename}'")
    
    plt.close()

def plot_pie_chart_no_text(counter, filename):
    """
    绘制不带标签的环形饼图（纯净版本）
    
    生成简洁的环形饼图，不包含任何文字标签，
    仅通过颜色区分不同类别，适合需要简洁展示的场景。
    
    参数:
        counter: Counter对象，包含要可视化的数据
        filename: 保存文件的路径
    """
    # 获取按计数排序的数据
    labels, values = zip(*counter.most_common())
    
    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 创建图表
    plt.figure(figsize=(10, 10), facecolor='white')
    ax = plt.subplot(111)
    
    # 为不同问题类型定义专门的颜色（与带标签版本一致）
    color_mapping = {
        'Linear programming': '#2E8B8B',      # 蓝绿色
        'Integer programming': '#2E318B',     # 深蓝色
        'Mixed integer programming': '#8B1A1A'  # 深红色
    }
    
    # 将颜色映射到实际标签
    colors = [color_mapping.get(label, '#999999') for label in labels]
    
    # 创建饼图（无标签，无百分比）
    ax.pie(
        values, 
        colors=colors,
        startangle=90,  # 从顶部开始
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True},
        labels=None,
        autopct=None
    )
    
    # 在中心绘制白色圆形，创建环形效果
    centre_circle = plt.Circle((0, 0), 0.5, fc='white', edgecolor='none')
    ax.add_patch(centre_circle)
    
    # 确保饼图为圆形
    ax.axis('equal')
    
    # 移除所有边框和刻度
    ax.set_frame_on(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved as '{filename}'")
    
    plt.close()

# ============================================================================
# 主程序
# ============================================================================

def main():
    """
    主函数：执行完整的分析和可视化流程
    
    步骤：
    1. 读取并分析JSON数据集
    2. 打印统计信息到控制台
    3. 生成水平条形图（类型分布）
    4. 生成饼图（问题分布，带标签和不带标签两个版本）
    """
    # 数据文件路径
    file_path = 'data/datasets/dataset_combined_result_mark.json'
    
    # 分析数据
    types_counter, problems_counter = analyze_json_data(file_path)
    
    # 打印统计信息
    print_counts(types_counter, "Type")
    print_counts(problems_counter, "Problem")
    
    # 生成类型分布的水平条形图
    plot_horizontal_bar(types_counter, 'Percentage of Cases by Problem Type', 'data/images/types_distribution.png')
    
    # 生成问题分布的饼图（带标签）
    plot_pie_chart(problems_counter, 'Problem Distribution', 'data/images/problems_distribution_pie.png')
    
    # 生成问题分布的饼图（不带标签）
    plot_pie_chart_no_text(problems_counter, 'data/images/problems_distribution_pie_no_text.png')
    
    print("\nTo view the charts, open the PNG files in your file browser.")

if __name__ == "__main__":
    main()
