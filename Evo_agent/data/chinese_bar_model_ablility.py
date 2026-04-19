"""
模型能力对比可视化工具
======================
本脚本用于比较不同LLM模型在两个基准测试上的表现：
1. LiveCodeBench (24/8.1–25/5) - 代码能力测试
2. AIME 2024 - 数学推理能力测试

生成分组条形图，展示每个模型在两个测试集上的得分，
使用不同颜色和填充图案区分不同公司的模型（OpenAI、Google、DeepSeek）。
"""

# ============================================================================
# 导入依赖库
# ============================================================================
import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算库
from matplotlib.patches import Patch  # 用于创建自定义图例

# ============================================================================
# 数据定义
# ============================================================================

# 模型名称列表（按照期望的显示顺序）
models = [
    "",  # OpenAI系列
    "",  # Google系列
    "",  # DeepSeek系列
    "",  # OpenAI系列
    "",  # OpenAI系列
    "",  # Google系列
    ""  # DeepSeek系列
]

# 两个基准测试的得分数据（%）
livecode_scores = [29.5, np.nan, 27.2, 75.8, 80.2, 73.6, 73.1]  # LiveCodeBench得分（np.nan表示无数据）
aime2024_scores = [11.7, 27.5, 25.0, 91.7, 89.2, 87.5, 70.0]  # AIME 2024得分

# ============================================================================
# 图表样式配置
# ============================================================================

# 两个数据集的名称
datasets = ["LiveCodeBench (24/8.1–25/5)", "AIME 2024"]
x = np.arange(len(datasets))  # x轴位置数组
width = 0.1  # 每个条形的宽度

# 颜色映射：根据公司/系列分配淡色
color_map = {
    "": "#d9d9d9",  # OpenAI系列1：浅灰色
    "": "#a6a6a6",  # OpenAI系列2：中灰色
    "": "#bfbfbf",  # OpenAI系列3：灰色
    "": "#b0e0dc",  # Google系列1：浅青色
    "": "#9dd8d4",  # Google系列2：青色
    "": "#a6c8ff",  # DeepSeek系列1：浅蓝色
    "": "#85b6ff"  # DeepSeek系列2：蓝色
}

# 填充图案映射：用于区分不同模型（适合黑白打印）
hatch_map = {
    "": "/",  # 斜线
    "": ".",  # 点
    "": "+",  # 加号
    "": "\\",  # 反斜线
    "": "o",  # 圆圈
    "": "x",  # X
    "": "*"  # 星号
}

# 设置全局字体为Times New Roman（学术论文标准）
plt.rcParams["font.family"] = "Times New Roman"

# ============================================================================
# 创建自定义图例
# ============================================================================

# 创建自定义图例句柄（大图标，保留填充图案）
legend_handles = [
    Patch(
        facecolor=color_map[model],  # 填充颜色
        edgecolor='white',  # 边框颜色
        hatch=hatch_map[model],  # 填充图案
        label=model,  # 标签
        linewidth=1.5  # 边框宽度
    )
    for model in models
]

# ============================================================================
# 绘制图表
# ============================================================================

# 创建图表
fig, ax = plt.subplots(figsize=(14, 6))

# 为每个模型绘制条形图
for i, model in enumerate(models):
    color = color_map[model]  # 获取该模型的颜色
    hatch = hatch_map[model]  # 获取该模型的填充图案
    scores = [livecode_scores[i], aime2024_scores[i]]  # 该模型在两个测试集上的得分
    
    # 计算该模型条形的x坐标位置
    x_positions = x + (i - len(models) / 2) * width + width / 2
    
    # 绘制条形图
    bars = ax.bar(
        x_positions,  # x坐标
        scores,  # 得分数据
        width,  # 条形宽度
        color=color,  # 填充颜色
        edgecolor='white',  # 边框颜色
        hatch=hatch,  # 填充图案
        linewidth=1.5  # 边框宽度
    )
    
    # 在每个条形上添加数值标签
    for bar in bars:
        height = bar.get_height()  # 获取条形高度（得分值）
        if not np.isnan(height):
            # 如果有数据，显示数值
            ax.annotate(
                f'{height:.1f}',  # 显示文字（保留一位小数）
                xy=(bar.get_x() + bar.get_width() / 2, height),  # 标注位置
                xytext=(0, 3),  # 文字偏移（向上3个点）
                textcoords="offset points",
                ha='center',  # 水平对齐：居中
                va='bottom',  # 垂直对齐：底部对齐
                fontsize=26  # 字体大小
            )
        else:
            # 如果无数据（NaN），显示"N/A"
            ax.annotate(
                'N/A',  # 显示"N/A"
                xy=(bar.get_x() + bar.get_width() / 2, 5),  # 固定位置（y=5）
                xytext=(0, 0),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=26,
                fontweight='bold'  # 粗体
            )

# ============================================================================
# 坐标轴和网格设置
# ============================================================================

# 设置y轴标签
ax.set_ylabel("Scores (%)", fontsize=28)
ax.tick_params(axis='y', labelsize=28)

# 设置x轴刻度和标签
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=28)

# 设置y轴范围
ax.set_ylim(0, 100)  # 0-100%

# 添加y轴网格线（虚线，半透明）
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 移除顶部和右侧边框线（更简洁的外观）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ============================================================================
# 图例设置
# ============================================================================

# 图例（两行，顶部居中，无边框）
ax.legend(
    handles=legend_handles,  # 使用自定义图例句柄
    loc='upper center',  # 位置：上方居中
    bbox_to_anchor=(0.5, 1.4),  # 具体位置微调（留出更多空间）
    ncol=4,  # 4列布局
    frameon=False,  # 无边框
    prop={'size': 20, 'family': 'Times New Roman'},  # 字体设置
    handlelength=2.5,  # 图例标记长度
    handleheight=1.5  # 图例标记高度
)

# ============================================================================
# 保存和显示
# ============================================================================

plt.tight_layout()  # 自动调整布局
plt.savefig("data/images/bar_model_ablility.png", dpi=300, bbox_inches='tight')  # 保存高清图片
plt.show()  # 显示图表

