import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("WorkerAssignment")

# 集合定义
I = ['I', 'II', 'III', 'IV', 'V']  # 工人集合
J = ['A', 'B', 'C', 'D']            # 任务集合

# 参数数据：工作时间表
c = {
    ('I', 'A'): 9, ('I', 'B'): 4, ('I', 'C'): 3, ('I', 'D'): 7,
    ('II', 'A'): 4, ('II', 'B'): 6, ('II', 'C'): 5, ('II', 'D'): 6,
    ('III', 'A'): 5, ('III', 'B'): 4, ('III', 'C'): 7, ('III', 'D'): 5,
    ('IV', 'A'): 7, ('IV', 'B'): 5, ('IV', 'C'): 2, ('IV', 'D'): 3,
    ('V', 'A'): 10, ('V', 'B'): 6, ('V', 'C'): 7, ('V', 'D'): 4
}

# 创建决策变量
x = {}
for i in I:
    for j in J:
        x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

# 设置目标函数：最小化总工作时间
model.setObjective(
    gp.quicksum(c[i, j] * x[i, j] for i in I for j in J),
    GRB.MINIMIZE
)

# 添加约束条件
# 每个任务必须分配给恰好一个工人
for j in J:
    model.addConstr(
        gp.quicksum(x[i, j] for i in I) == 1,
        name=f"Task_{j}_assigned"
    )

# 每个工人最多被分配一个任务
for i in I:
    model.addConstr(
        gp.quicksum(x[i, j] for j in J) <= 1,
        name=f"Worker_{i}_limit"
    )

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最优总工作时间: {model.ObjVal}")
    print("分配方案:")
    for i in I:
        for j in J:
            if x[i, j].X > 0.5:
                print(f"工人 {i} -> 任务 {j} (工时: {c[i, j]})")
else:
    print("未找到最优解")