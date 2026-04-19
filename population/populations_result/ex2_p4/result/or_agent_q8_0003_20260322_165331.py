import gurobipy as gp
from gurobipy import GRB

# 定义集合和参数
workers = [1, 2, 3, 4, 5]  # 工人 I, II, III, IV, V 对应 1,2,3,4,5
tasks = ['A', 'B', 'C', 'D']

# 成本参数 c_ij，基于表格5-2
cost = {
    (1, 'A'): 9, (1, 'B'): 4, (1, 'C'): 3, (1, 'D'): 7,
    (2, 'A'): 4, (2, 'B'): 6, (2, 'C'): 5, (2, 'D'): 6,
    (3, 'A'): 5, (3, 'B'): 4, (3, 'C'): 7, (3, 'D'): 5,
    (4, 'A'): 7, (4, 'B'): 5, (4, 'C'): 2, (4, 'D'): 3,
    (5, 'A'): 10, (5, 'B'): 6, (5, 'C'): 7, (5, 'D'): 4
}

# 创建模型
model = gp.Model("WorkerTaskAssignment")

# 定义决策变量
x = model.addVars(workers, tasks, vtype=GRB.BINARY, name="x")

# 设置目标函数：最小化总工作时间
model.setObjective(
    gp.quicksum(cost[i, j] * x[i, j] for i in workers for j in tasks),
    GRB.MINIMIZE
)

# 添加约束条件
# 每个任务必须分配给一个工人
for j in tasks:
    model.addConstr(
        gp.quicksum(x[i, j] for i in workers) == 1,
        name=f"Task_{j}_assigned"
    )

# 每个工人最多分配一个任务
for i in workers:
    model.addConstr(
        gp.quicksum(x[i, j] for j in tasks) <= 1,
        name=f"Worker_{i}_limit"
    )

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最优总时间: {model.objVal} 小时")
    print("分配方案:")
    for i in workers:
        for j in tasks:
            if x[i, j].x > 0.5:
                print(f"  工人 {i} -> 任务 {j} (时间: {cost[i, j]}小时)")
else:
    print("未找到最优解")