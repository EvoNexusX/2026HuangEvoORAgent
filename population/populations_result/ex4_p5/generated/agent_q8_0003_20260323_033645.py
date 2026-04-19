import gurobipy as gp
from gurobipy import GRB

# Step 1: 环境初始化与数据准备
workers = ['I', 'II', 'III', 'IV', 'V']
tasks = ['A', 'B', 'C', 'D']

cost = {
    ('I', 'A'): 9, ('I', 'B'): 4, ('I', 'C'): 3, ('I', 'D'): 7,
    ('II', 'A'): 4, ('II', 'B'): 6, ('II', 'C'): 5, ('II', 'D'): 6,
    ('III', 'A'): 5, ('III', 'B'): 4, ('III', 'C'): 7, ('III', 'D'): 5,
    ('IV', 'A'): 7, ('IV', 'B'): 5, ('IV', 'C'): 2, ('IV', 'D'): 3,
    ('V', 'A'): 10, ('V', 'B'): 6, ('V', 'C'): 7, ('V', 'D'): 4
}

# Step 2: 创建模型
model = gp.Model("Assignment_Problem")

# Step 3: 定义决策变量
x = model.addVars(workers, tasks, vtype=GRB.BINARY, name="x")

# Step 4: 设置目标函数
model.setObjective(
    gp.quicksum(cost[i, j] * x[i, j] for i in workers for j in tasks),
    GRB.MINIMIZE
)

# Step 5: 添加约束条件
# 每个任务必须由一个工人完成
for j in tasks:
    model.addConstr(
        gp.quicksum(x[i, j] for i in workers) == 1,
        name=f"Task_{j}_assigned"
    )

# 每个工人最多完成一个任务
for i in workers:
    model.addConstr(
        gp.quicksum(x[i, j] for j in tasks) <= 1,
        name=f"Worker_{i}_limit"
    )

# Step 6: 模型求解
model.optimize()

# Step 7: 结果提取与输出
if model.status == GRB.OPTIMAL:
    print("最优解找到。")
    print(f"最小总工作时间: {model.ObjVal}")
    print("\n具体分配方案:")
    for i in workers:
        for j in tasks:
            if x[i, j].X > 0.5:
                print(f"工人 {i} -> 任务 {j} (时间: {cost[i, j]})")
else:
    print(f"未找到最优解。求解状态: {model.status}")