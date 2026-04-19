from gurobipy import Model, GRB, quicksum

# 定义问题数据
workers = [0, 1, 2, 3, 4]  # 0: I, 1: II, 2: III, 3: IV, 4: V
tasks = [0, 1, 2, 3]       # 0: A, 1: B, 2: C, 3: D
cost = [
    [9, 4, 3, 7],   # 工人I
    [4, 6, 5, 6],   # 工人II
    [5, 4, 7, 5],   # 工人III
    [7, 5, 2, 3],   # 工人IV
    [10, 6, 7, 4]   # 工人V
]

# 创建模型
model = Model("Assignment_Problem")

# 添加决策变量
x = model.addVars(workers, tasks, vtype=GRB.BINARY, name="x")

# 设置目标函数：最小化总工作时间
model.setObjective(quicksum(cost[i][j] * x[i, j] for i in workers for j in tasks), GRB.MINIMIZE)

# 添加约束条件：每个任务必须恰好由一名工人完成
for j in tasks:
    model.addConstr(quicksum(x[i, j] for i in workers) == 1, name=f"task_{j}")

# 添加约束条件：每名工人最多完成一项任务
for i in workers:
    model.addConstr(quicksum(x[i, j] for j in tasks) <= 1, name=f"worker_{i}")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最优总时间: {model.objVal}")
    print("分配方案:")
    for i in workers:
        for j in tasks:
            if x[i, j].X > 0.5:
                worker_name = ['I', 'II', 'III', 'IV', 'V'][i]
                task_name = ['A', 'B', 'C', 'D'][j]
                print(f"工人 {worker_name} -> 任务 {task_name} (时间: {cost[i][j]})")
else:
    print("未找到最优解")