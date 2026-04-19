"""
推理模型与基础模型性能差异对比可视化工具
============================================
本脚本用于比较推理模型（Reasoning Models）与基础模型（Base Models）
在多个数据集上的性能差异。

比较的模型对包括：
1.  vs 
2.  vs   
3.  vs 
4.  vs 

生成分组条形图，展示每对模型在5个数据集上的性能差异百分比。
"""

# ============================================================================
# 导入依赖库
# ============================================================================
import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算库
import pandas as pd  # 数据处理库

# ============================================================================
# 数据定义
# ============================================================================

# 5个数据集的名称
datasets = ['IndustryOR', 'ComplexLP', 'EasyLP', 'NL4OPT', 'BWOR']

# 4对模型的对比（推理模型 - 基础模型）
models = [
    ' - ',  # OpenAI推理模型对比
    ' - ',  # OpenAI轻量推理模型对比
    ' - ',  # Google推理模型对比
    ' - '  # DeepSeek推理模型对比
]

# 性能差异数据（%）
# 正值表示推理模型更好，负值表示基础模型更好
data = [
    [1.00, 3.32, -14.11, 5.31, 35.37],  #  - 
    [-5.00, 3.32, -1.84, 6.53, 32.93],  #  - 
    [3.00, -7.11, -17.79, -4.90, 21.95],  #  - 
    [-1.00, 5.69, -8.28, -1.63, 10.98]  #  - 
]

# 使用pandas DataFrame组织数据（方便数据操作）
df = pd.DataFrame(data, index=models, columns=datasets)

# ============================================================================
# 图表样式配置
# ============================================================================

# x轴位置设置
x = np.arange(len(datasets))  # 数据集数量对应的x坐标数组
bar_width = 0.2  # 每个条形的宽度

# 四对模型的颜色（灰色系和蓝色系）
colors = [
    'lightgray',  #  - ：浅灰色
    'dimgray',  #  - ：深灰色
    'powderblue',  #  - ：粉蓝色
    'cornflowerblue'  #  - ：矢车菊蓝
]

# 四对模型的填充图案（用于区分，适合黑白打印）
hatches = [
    '///',  # 密集斜线
    '...',  # 密集点
    'oo',  # 圆圈
    '**'  # 星号
]

# 设置全局字体为Times New Roman（学术论文标准）
plt.rcParams['font.family'] = 'Times New Roman'

# ============================================================================
# 绘制图表
# ============================================================================

# 创建图表
fig, ax = plt.subplots(figsize=(14, 6))

# 为每对模型绘制条形图
for i, (model, color, hatch) in enumerate(zip(models, colors, hatches)):
    # 绘制条形图
    bars = ax.bar(
        x + i * bar_width,  # x坐标位置（每对模型偏移一个bar_width）
        df.loc[model],  # 该模型对在各数据集上的差异值
        width=bar_width,  # 条形宽度
        label=model,  # 图例标签
        color=color,  # 填充颜色
        hatch=hatch,  # 填充图案
        edgecolor='white'  # 边框颜色：白色
    )
    
    # 在每个条形上添加数值标签
    for bar in bars:
        height = bar.get_height()  # 获取条形高度（差异值）
        # 根据正负值决定标签位置
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x坐标（条形中心）
            height + 0.5 if height >= 0 else height - 2,  # y坐标（正值在上方，负值在下方）
            f'{height:.2f}',  # 显示文字（保留两位小数）
            ha='center',  # 水平对齐：居中
            va='bottom' if height >= 0 else 'top',  # 垂直对齐：正值底部，负值顶部
            fontsize=18  # 字体大小
        )

# ============================================================================
# 坐标轴和网格设置
# ============================================================================

# 设置y轴标签
ax.set_ylabel('Performance Difference (%)', fontsize=28)
ax.tick_params(axis='y', labelsize=28)

# 设置x轴刻度和标签
ax.set_xticks(x + 1.5 * bar_width)  # x轴刻度位置（每组中心）
ax.set_xticklabels(datasets, fontsize=28)  # x轴标签（数据集名称）

# 添加y=0的水平参考线（黑色实线）
ax.axhline(0, color='black', linewidth=0.8)

# 设置y轴范围
ax.set_ylim(-25, 40)  # -25%到40%

# 添加y轴网格线（虚线，半透明）
ax.grid(axis='y', linestyle='--', alpha=0.5)

# ============================================================================
# 图例设置
# ============================================================================

# 图例置顶（位于图表上方，2列布局）
ax.legend(
    title="",  # 无标题
    loc="upper center",  # 位置：上方居中
    bbox_to_anchor=(0.5, 1.35),  # 具体位置微调
    ncol=2,  # 2列布局
    fontsize=28,  # 字体大小
    handlelength=2.5,  # 图例标记长度
    columnspacing=1.5,  # 列间距
    frameon=False  # 无边框
)

# 移除顶部和右侧边框线（更简洁的外观）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ============================================================================
# 保存和显示
# ============================================================================

plt.tight_layout()  # 自动调整布局
plt.savefig("data/images/bar_model_compare.png", dpi=300, bbox_inches='tight')  # 保存高清图片
plt.show()  # 显示图表

