import gurobipy as gp
from gurobipy import GRB

# 定义工人和任务集合
workers = ['I', 'II', 'III', 'IV', 'V']
tasks = ['A', 'B', 'C', 'D']

# 定义工作时间参数
cost = {
    ('I', 'A'): 9, ('I', 'B'): 4, ('I', 'C'): 3, ('I', 'D'): 7,
    ('II', 'A'): 4, ('II', 'B'): 6, ('II', 'C'): 5, ('II', 'D'): 6,
    ('III', 'A'): 5, ('III', 'B'): 4, ('III', 'C'): 7, ('III', 'D'): 5,
    ('IV', 'A'): 7, ('IV', 'B'): 5, ('IV', 'C'): 2, ('IV', 'D'): 3,
    ('V', 'A'): 10, ('V', 'B'): 6, ('V', 'C'): 7, ('V', 'D'): 4
}

# 创建模型
model = gp.Model("WorkerAssignment")

# 添加决策变量
x = model.addVars(workers, tasks, vtype=GRB.BINARY, name="x")

# 设置目标函数：最小化总工作时间
model.setObjective(gp.quicksum(cost[i, j] * x[i, j] for i in workers for j in tasks), GRB.MINIMIZE)

# 添加约束：每个任务必须恰好由一个工人完成
for j in tasks:
    model.addConstr(gp.quicksum(x[i, j] for i in workers) == 1, name=f"Task_{j}")

# 添加约束：每个工人最多分配一个任务
for i in workers:
    model.addConstr(gp.quicksum(x[i, j] for j in tasks) <= 1, name=f"Worker_{i}")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最小总工作时间: {model.ObjVal}")
    print("分配方案:")
    for i in workers:
        for j in tasks:
            if x[i, j].X > 0.5:
                print(f"工人 {i} -> 任务 {j}")
else:
    print("未找到最优解")