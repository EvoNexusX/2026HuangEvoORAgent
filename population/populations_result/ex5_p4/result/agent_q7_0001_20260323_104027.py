from gurobipy import *

# 定义仓库和港口的索引列表
warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]
I = range(len(warehouses))  # 0,1,2,3,4,5
J = range(len(ports))       # 0,1,2,3,4

# 供应量和需求量
supply = [10, 12, 20, 24, 18, 40]
demand = [20, 15, 25, 33, 21]

# 距离矩阵（公里）
dist = [
    [290, 115, 355, 715, 810],
    [380, 340, 165, 380, 610],
    [505, 530, 285, 220, 450],
    [655, 450, 155, 240, 315],
    [1010, 840, 550, 305, 95],
    [1072, 1097, 747, 372, 333]
]

# 计算成本矩阵：每集装箱每公里30欧元
cost = [[30 * dist[i][j] for j in J] for i in I]

# 创建模型
model = Model("Transportation")

# 添加变量：从仓库i到港口j的集装箱数量，连续非负
x = model.addVars(I, J, lb=0.0, name="x")

# 设置目标函数：最小化总运输成本
model.setObjective(quicksum(cost[i][j] * x[i, j] for i in I for j in J), GRB.MINIMIZE)

# 添加供应约束：每个仓库运出总量不超过供应量
model.addConstrs((x.sum(i, '*') <= supply[i] for i in I), name="Supply")

# 添加需求约束：每个港口运入总量等于需求量
model.addConstrs((x.sum('*', j) == demand[j] for j in J), name="Demand")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最优总成本: {model.objVal:.2f} 欧元")
    print("\n运输方案（非零的运输量）:")
    for i in I:
        for j in J:
            if x[i, j].x > 1e-6:
                print(f"{warehouses[i]} -> {ports[j]}: {x[i, j].x:.0f} 个集装箱")
else:
    print("未找到最优解")