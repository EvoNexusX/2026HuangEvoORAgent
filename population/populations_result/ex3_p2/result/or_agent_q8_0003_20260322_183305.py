import gurobipy as gp
from gurobipy import GRB

# 初始化模型
model = gp.Model("Worker_Task_Assignment")

# 工人集合
workers = ['I', 'II', 'III', 'IV', 'V']

# 任务集合
tasks = ['A', 'B', 'C', 'D']

# 工作时间矩阵 (workers × tasks)
cost_matrix = {
    'I': {'A': 9, 'B': 4, 'C': 3, 'D': 7},
    'II': {'A': 4, 'B': 6, 'C': 5, 'D': 6},
    'III': {'A': 5, 'B': 4, 'C': 7, 'D': 5},
    'IV': {'A': 7, 'B': 5, 'C': 2, 'D': 3},
    'V': {'A': 10, 'B': 6, 'C': 7, 'D': 4}
}

# 创建二进制决策变量字典
x = model.addVars(
    workers, 
    tasks, 
    vtype=GRB.BINARY,
    name="x"
)

# 设置目标函数：最小化总工作小时数
model.setObjective(
    gp.quicksum(
        cost_matrix[i][j] * x[i, j] 
        for i in workers 
        for j in tasks
    ),
    GRB.MINIMIZE
)

# 添加约束：每个任务必须恰好分配给一个工人
for j in tasks:
    model.addConstr(
        gp.quicksum(x[i, j] for i in workers) == 1,
        name=f"Task_{j}_Coverage"
    )

# 添加约束：每个工人最多分配一个任务
for i in workers:
    model.addConstr(
        gp.quicksum(x[i, j] for j in tasks) <= 1,
        name=f"Worker_{i}_Capacity"
    )

# 求解模型
model.optimize()

# 检查求解状态并输出结果
if model.status == GRB.OPTIMAL:
    print(f"最优目标值（总工作时间）: {model.objVal}")
    print("\n分配方案:")
    assignment = {}
    for i in workers:
        for j in tasks:
            if x[i, j].X > 0.5:  # 判断变量是否为1
                assignment[j] = i
                print(f"  任务 {j} → 工人 {i}")
    
    # 识别未分配的工人
    assigned_workers = set(assignment.values())
    unassigned = [i for i in workers if i not in assigned_workers]
    if unassigned:
        print(f"\n未分配的工人: {', '.join(unassigned)}")
else:
    print("未找到最优解")