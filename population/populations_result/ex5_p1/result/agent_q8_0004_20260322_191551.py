import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("WorkerAssignment")

# 定义集合
I = [1, 2, 3, 4, 5]  # 工人I,II,III,IV,V
J = ['A', 'B', 'C', 'D']  # 任务A,B,C,D

# 定义参数（工作时间）
c = {
    (1, 'A'): 9, (1, 'B'): 4, (1, 'C'): 3, (1, 'D'): 7,
    (2, 'A'): 4, (2, 'B'): 6, (2, 'C'): 5, (2, 'D'): 6,
    (3, 'A'): 5, (3, 'B'): 4, (3, 'C'): 7, (3, 'D'): 5,
    (4, 'A'): 7, (4, 'B'): 5, (4, 'C'): 2, (4, 'D'): 3,
    (5, 'A'):10, (5, 'B'): 6, (5, 'C'): 7, (5, 'D'): 4
}

# 添加决策变量
x = model.addVars(I, J, vtype=GRB.BINARY, name="x")

# 设置目标函数：最小化总工作时间
model.setObjective(gp.quicksum(c[i, j] * x[i, j] for i in I for j in J), GRB.MINIMIZE)

# 添加约束条件：每个任务必须由恰好一个工人完成
for j in J:
    model.addConstr(gp.quicksum(x[i, j] for i in I) == 1, f"Task_{j}_Assignment")

# 添加约束条件：每个工人最多被分配一个任务
for i in I:
    model.addConstr(gp.quicksum(x[i, j] for j in J) <= 1, f"Worker_{i}_Capacity")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最小总工作时间: {model.objVal} 小时")
    print("\n分配方案:")
    for i in I:
        for j in J:
            if x[i, j].X > 0.5:
                print(f"  工人 {i} -> 任务 {j} (时间: {c[i, j]} 小时)")
else:
    print("未找到最优解")