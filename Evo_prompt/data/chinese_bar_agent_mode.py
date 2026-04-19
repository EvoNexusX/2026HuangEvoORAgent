"""
Agent模式性能比较可视化工具
============================
本脚本用于可视化比较不同LLM模型在三种Agent模式下的性能表现：
1. Direct Code Generation（直接代码生成）
2. Math Agent + Code Agent（数学建模 + 代码生成）
3. Math Agent + Code Agent + Debugging Agent（数学建模 + 代码生成 + 调试）

生成分组水平条形图，展示每个模型在不同模式下的准确率，
并通过箭头标注Agent模式带来的性能提升。
"""

# ============================================================================
# 导入依赖库
# ============================================================================
import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算库
from matplotlib.patches import Rectangle  # 用于创建空白图例条目

# ============================================================================
# 数据定义
# ============================================================================

# 模型列表（从平均值到具体模型）
models = [
    "Average",  # 平均值
    "", 
    "", 
    "", 
    "", 
    "", 
    ""
]

# 三种Agent方法
methods = [
    "Direct Code Generation",  # 方法1：直接生成代码
    "Math Agent + Code Agent",  # 方法2：数学建模 + 代码生成
    "Math Agent + Code\n Agent + Debugging Agent"  # 方法3：数学建模 + 代码生成 + 调试
]

# 每个模型在三种方法下的准确率得分（%）
# 每行代表一个模型，每列代表一种方法
original_scores = [
    [79.27, 75.61, 75.61],  # 
    [80.49, 73.17, 71.95],  # 
    [82.93, 75.61, 73.17],  # 
    [52.44, 45.12, 40.24],  # 
    [65.85, 62.20, 50.00],  # 
    [69.51, 63.41, 62.20]   # 
]

# 计算每种方法的平均准确率
averages = []
for i in range(3):  # 3种方法
    # 对每种方法，计算所有模型的平均值
    method_avg = sum(score[i] for score in original_scores) / len(original_scores)
    averages.append(round(method_avg, 2))

# 将平均值添加到scores开头，然后反转整个列表
# 反转是为了让最好的模型显示在图表顶部
scores = [averages] + original_scores[::-1]

# ============================================================================
# 图表样式配置
# ============================================================================

# 三种方法的颜色
colors = ['lightgray',  # 方法1：浅灰色
          'lightblue',  # 方法2：浅蓝色  
          'darkgray']   # 方法3：深灰色

# 三种方法的填充图案（用于区分）
hatches = ['//',  # 方法1：斜线
           '*',   # 方法2：星号
           '.']   # 方法3：点

# 设置全局字体为Times New Roman（学术论文标准）
plt.rcParams["font.family"] = "Times New Roman"

# ============================================================================
# 绘制图表
# ============================================================================

# 图表布局参数
group_spacing = 2.0  # 模型组之间的间距
y = np.arange(len(models)) * group_spacing  # 每个模型组的y坐标
total_width = 1.8  # 每组条形的总宽度
bar_width = total_width / len(methods)  # 单个条形的宽度

# 创建图表（尺寸较大以容纳更多信息）
fig, ax = plt.subplots(figsize=(14, 12))

# 为每种方法绘制条形图
for i, method in enumerate(methods):
    # 使用反向索引来绘制，以保持图例顺序正确
    reverse_i = len(methods) - 1 - i
    # 提取该方法下所有模型的得分
    method_scores = [scores[j][reverse_i] for j in range(len(models))]
    
    # 绘制水平条形图
    bars = ax.barh(
        y + reverse_i * bar_width,  # y坐标位置
        method_scores,  # 条形长度（得分）
        height=bar_width,  # 条形高度
        label=method,  # 图例标签
        color=colors[i],  # 填充颜色
        hatch=hatches[i],  # 填充图案
        edgecolor='white',  # 边框颜色
        linewidth=1.5  # 边框宽度
    )
    
    # 在每个条形内部添加数值标签
    for bar in bars:
        width = bar.get_width()  # 获取条形宽度（得分值）
        # 在条形右侧内部添加文字
        ax.text(
            width - 3,  # x坐标（条形右侧留出3个单位）
            bar.get_y() + bar.get_height() / 2,  # y坐标（条形垂直居中）
            f'{width:.2f}',  # 显示文字（保留两位小数）
            ha='right',  # 水平对齐：右对齐
            va='center',  # 垂直对齐：居中
            fontsize=20,  # 字体大小
            color='black',  # 字体颜色
            weight='bold'  # 字体粗细：粗体
        )

# ============================================================================
# 坐标轴和网格设置
# ============================================================================

# 设置x轴标签
ax.set_xlabel("Accuracy (%)", fontsize=40)

# 设置y轴刻度位置和标签
ax.set_yticks(y + bar_width * (len(methods) - 1) / 2)  # y轴刻度位置（每组中心）
ax.set_yticklabels(models, fontsize=40)  # y轴标签（模型名称）

# 设置x轴刻度字体大小
ax.tick_params(axis='x', labelsize=40)

# 设置x轴和y轴范围
ax.set_xlim(0, 90)  # x轴：0-90%
ax.set_ylim(-0.5, y[-1] + total_width + 0.5)  # y轴：容纳所有条形

# 添加x轴网格线（虚线，半透明）
ax.grid(axis='x', linestyle='--', alpha=0.5)

# 移除顶部和右侧边框线（更简洁的外观）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ============================================================================
# 添加性能提升箭头和标签
# ============================================================================

# 计算Average组各方法条形的中心y坐标
avg_group_y = y[0]  # Average组的基准y坐标

# 三种方法的条形中心y坐标
direct_code_y_center = avg_group_y + 2 * bar_width  # Direct Code Generation
math_agent_code_y_center = avg_group_y + 1 * bar_width  # Math Agent + Code Agent
math_agent_debug_y_center = avg_group_y + 0 * bar_width  # Math Agent + Code Agent + Debugging Agent

# 三种方法的准确率值（从averages中获取）
direct_code_x = 62.20  # Direct Code Generation的值
math_agent_code_x = 66.26  # Math Agent + Code Agent的值
math_agent_debug_x = 71.75  # Math Agent + Code Agent + Debugging Agent的值

# 添加第一个弯曲箭头：从Direct Code到Math Agent + Code
ax.annotate(
    '',  # 不添加文字，只画箭头
    xy=(math_agent_code_x, math_agent_code_y_center),  # 箭头终点
    xytext=(direct_code_x, direct_code_y_center),  # 箭头起点
    arrowprops=dict(
        arrowstyle='->',  # 箭头样式
        connectionstyle='arc3,rad=-0.3',  # 连接样式：弧形，弯曲度
        color='red',  # 红色
        lw=2  # 线宽
    )
)

# 添加"4.06%"标签（表示性能提升）
arrow_mid_x = direct_code_x + 10  # 标签x坐标（箭头中间偏右）
arrow_mid_y = (direct_code_y_center + math_agent_code_y_center) / 2  # 标签y坐标（两条形中间）
ax.text(
    arrow_mid_x, arrow_mid_y, '4.06%',  # 位置和文字
    fontsize=22, color='red',  # 样式
    weight='bold', ha='center', va='center'  # 对齐方式
)

# 添加第二个弯曲箭头：从Math Agent + Code到Math Agent + Code + Debugging
ax.annotate(
    '',
    xy=(math_agent_debug_x, math_agent_debug_y_center),  # 箭头终点
    xytext=(math_agent_code_x, math_agent_code_y_center),  # 箭头起点
    arrowprops=dict(
        arrowstyle='->',
        connectionstyle='arc3,rad=-0.3',
        color='blue',  # 蓝色
        lw=2
    )
)

# 添加"5.49%"标签
arrow2_mid_x = math_agent_code_x + 10
arrow2_mid_y = (math_agent_code_y_center + math_agent_debug_y_center) / 2
ax.text(
    arrow2_mid_x, arrow2_mid_y, '5.49%',
    fontsize=22, color='blue',
    weight='bold', ha='center', va='center'
)

# ============================================================================
# 图例设置
# ============================================================================

# 获取当前图例句柄和标签
handles, labels = ax.get_legend_handles_labels()

# 创建空白图例条目（用于调整图例布局）
blank_handle = Rectangle((0,0), 0, 0, fill=False, edgecolor='none', visible=False)

# 重新排列图例：
# 第一行：方法1, 方法2
# 第二行：空白, 方法3
legend = ax.legend(
    [handles[0], handles[1], blank_handle, handles[2]],  # 图例句柄
    [labels[0], labels[1], ' ', labels[2]],  # 图例标签
    title="",  # 无标题
    loc="upper center",  # 位置：上方居中
    bbox_to_anchor=(0.35, 1.12),  # 具体位置微调
    ncol=3,  # 3列布局
    fontsize=24,  # 字体大小
    handlelength=2.5,  # 图例标记长度
    columnspacing=0.5,  # 列间距
    frameon=False  # 无边框
)

# ============================================================================
# 保存和显示
# ============================================================================

plt.tight_layout()  # 自动调整布局
plt.savefig("data/images/bar_agent_mode.png", dpi=300, bbox_inches='tight')  # 保存高清图片
plt.show()  # 显示图表

