import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("Assignment")

# 定义工人和任务集合
workers = ['I', 'II', 'III', 'IV', 'V']
tasks = ['A', 'B', 'C', 'D']

# 定义工作时间成本，数据来自表5-2
cost = {
    ('I', 'A'): 9, ('I', 'B'): 4, ('I', 'C'): 3, ('I', 'D'): 7,
    ('II', 'A'): 4, ('II', 'B'): 6, ('II', 'C'): 5, ('II', 'D'): 6,
    ('III', 'A'): 5, ('III', 'B'): 4, ('III', 'C'): 7, ('III', 'D'): 5,
    ('IV', 'A'): 7, ('IV', 'B'): 5, ('IV', 'C'): 2, ('IV', 'D'): 3,
    ('V', 'A'): 10, ('V', 'B'): 6, ('V', 'C'): 7, ('V', 'D'): 4
}

# 创建决策变量
x = model.addVars(workers, tasks, vtype=GRB.BINARY, name="x")

# 设置目标函数：最小化总工作时间
model.setObjective(gp.quicksum(cost[w, t] * x[w, t] for w in workers for t in tasks), GRB.MINIMIZE)

# 添加约束：每个任务必须由恰好一名工人完成
model.addConstrs((x.sum('*', t) == 1 for t in tasks), name="TaskAssignment")

# 添加约束：每名工人最多完成一项任务
model.addConstrs((x.sum(w, '*') <= 1 for w in workers), name="WorkerLimit")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最小总工作时间: {model.ObjVal:.0f}")
    print("分配方案:")
    for w in workers:
        for t in tasks:
            if x[w, t].X > 0.5:
                print(f"  工人 {w} -> 任务 {t}")
else:
    print(f"求解失败，状态: {model.status}")