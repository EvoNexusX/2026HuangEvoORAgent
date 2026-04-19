"""
问题长度分析与可视化工具
========================
本脚本用于分析数据集中问题文本的长度分布，并生成可视化图表。

主要功能：
1. 统计所有问题的字符长度
2. 生成长度分布直方图
3. 叠加核密度估计曲线（KDE）
4. 显示统计信息（最小/最大/平均/中位数长度）
5. 保存结果为高质量PNG图片

输出文件：
- question_lengths.txt: 所有问题长度的列表
- question_length_histogram.png: 长度分布可视化图表

可视化特点：
- 使用Times New Roman字体（学术风格）
- 100字符为区间的直方图
- 叠加平滑的KDE曲线
- 显示详细统计信息
"""

# ============================================================================
# 导入依赖库
# ============================================================================
import json  # JSON数据处理
import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算
from scipy import stats  # 统计分析（用于KDE）

# ============================================================================
# 数据加载与长度统计
# ============================================================================

# JSON数据文件路径
json_file_path = "data/datasets/dataset_combined_result_mark.json"

# 读取JSON文件
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 存储问题长度的列表
question_lengths = []

# 遍历JSON中的每个条目
for key, item in data.items():
    if "question" in item:
        # 计算问题文本的字符长度
        question_length = len(item["question"])
        question_lengths.append(question_length)

# 打印统计结果
print(f"Number of questions processed: {len(question_lengths)}")
print(f"Question lengths: {question_lengths}")

# 将问题长度列表保存到文本文件
with open("data/datasets/question_lengths.txt", 'w') as output_file:
    output_file.write(str(question_lengths))

# ============================================================================
# 可视化配置（学术出版风格）
# ============================================================================

# 设置绘图样式（seaborn白色网格风格）
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.facecolor'] = 'white'  # 白色背景
plt.rcParams['figure.facecolor'] = 'white'

# 设置字体为Times New Roman（学术风格）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16  # 基础字体大小
plt.rcParams['axes.titlesize'] = 24  # 标题字体大小
plt.rcParams['axes.labelsize'] = 20  # 轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18  # X轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18  # Y轴刻度字体大小

# ============================================================================
# 生成直方图与核密度估计曲线
# ============================================================================

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))

# 创建100字符为单位的区间（bin）
min_length = min(question_lengths)  # 最小长度
max_length = max(question_lengths)  # 最大长度
# 生成从最小值向下取整到100的倍数，到最大值向上取整到100的倍数的区间
bin_edges = np.arange(min_length - min_length % 100, max_length + 100, 100)

# 创建直方图（density=True用于归一化，便于与KDE比较）
histogram = ax.hist(question_lengths, bins=bin_edges.tolist(), alpha=1.0, 
                   color='#2E8B8B',  # 青色系
                   edgecolor='white',  # 白色边框
                   linewidth=0.5,  # 边框宽度
                   density=True)  # 归一化（概率密度）

# 生成平滑曲线的X值（比区间更密集的点）
x_values = np.linspace(min_length, max_length, 1000)

# 创建核密度估计（KDE）用于平滑曲线
# bw_method='scott'使用Scott准则自动选择带宽
kde = stats.gaussian_kde(question_lengths, bw_method='scott')
density_curve = kde(x_values)

# 绘制KDE曲线
ax.plot(x_values, density_curve, color='#2A2E8C',  # 深蓝色
        linewidth=2)

# ============================================================================
# 图表美化
# ============================================================================

# 移除顶部和右侧边框（更简洁）
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# 添加轴标签和标题
ax.set_xlabel('Question Length (characters)', fontsize=20)
ax.set_ylabel('Frequency Distribution', fontsize=20)
ax.set_title('Question Length Distribution', fontsize=24, color='#2E8B57')  # 海绿色标题

# ============================================================================
# 添加统计信息文本框
# ============================================================================

# 准备统计信息文本
stats_text = (f"Total Questions: {len(question_lengths)}\n"
            f"Min Length: {min_length}\n"
            f"Max Length: {max_length}\n"
            f"Average Length: {sum(question_lengths)/len(question_lengths):.2f}\n"
            f"Median Length: {np.median(question_lengths):.2f}")

# 在图表右上角显示统计信息
# transform=ax.transAxes表示使用相对坐标（0-1范围）
ax.text(0.73, 0.95, stats_text, 
       transform=ax.transAxes, 
       fontsize=16, 
       verticalalignment='top',  # 顶部对齐
       fontfamily='Times New Roman',
       bbox=dict(facecolor='white',  # 白色背景
                alpha=0.8,  # 半透明
                edgecolor='none',  # 无边框
                pad=5))  # 内边距

# ============================================================================
# 网格线和布局优化
# ============================================================================

# 只显示Y轴水平网格线（更简洁）
ax.yaxis.grid(True, linestyle='-', alpha=0.2)  # 淡色网格线
ax.xaxis.grid(False)  # 不显示X轴网格线

# Y轴从0开始（避免负值）
ax.set_ylim(bottom=0)

# 调整布局（避免标签被裁剪）
plt.tight_layout()

# ============================================================================
# 保存图表
# ============================================================================

# 保存为高分辨率PNG图片（300 DPI，适合出版）
plt.savefig('data/images/question_length_histogram.png', 
           dpi=300,  # 高分辨率
           bbox_inches='tight')  # 自动裁剪空白边缘

print("Histogram saved to data/images/question_length_histogram.png")
