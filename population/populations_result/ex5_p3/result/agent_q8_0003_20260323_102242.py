import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("WorkerAssignment")

# 数据定义
workers = [1, 2, 3, 4, 5]  # 对应工人 I, II, III, IV, V
tasks = ['A', 'B', 'C', 'D']

# 成本矩阵（工作时间）
cost = {
    (1, 'A'): 9, (1, 'B'): 4, (1, 'C'): 3, (1, 'D'): 7,
    (2, 'A'): 4, (2, 'B'): 6, (2, 'C'): 5, (2, 'D'): 6,
    (3, 'A'): 5, (3, 'B'): 4, (3, 'C'): 7, (3, 'D'): 5,
    (4, 'A'): 7, (4, 'B'): 5, (4, 'C'): 2, (4, 'D'): 3,
    (5, 'A'): 10, (5, 'B'): 6, (5, 'C'): 7, (5, 'D'): 4,
}

# 创建决策变量
x = model.addVars(workers, tasks, vtype=GRB.BINARY, name="x")

# 设置目标函数
model.setObjective(gp.quicksum(cost[i, j] * x[i, j] for i in workers for j in tasks), GRB.MINIMIZE)

# 添加约束条件
# 每个任务必须恰好由一个工人完成
for j in tasks:
    model.addConstr(gp.quicksum(x[i, j] for i in workers) == 1, f"Task_{j}")

# 每个工人最多承担一个任务
for i in workers:
    model.addConstr(gp.quicksum(x[i, j] for j in tasks) <= 1, f"Worker_{i}")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最优总工作时间: {model.objVal:.0f} 小时")
    print("分配方案:")
    for i in workers:
        for j in tasks:
            if x[i, j].X > 0.5:
                print(f"  工人 {i} -> 任务 {j} (耗时: {cost[i, j]} 小时)")
else:
    print("未找到最优解")