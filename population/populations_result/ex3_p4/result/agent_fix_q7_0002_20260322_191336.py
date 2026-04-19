# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

# 定义数据
warehouses = ['Verona', 'Perugia', 'Rome', 'Pescara', 'Taranto', 'Lamezia']
ports = ['Genoa', 'Venice', 'Ancona', 'Naples', 'Bari']

supply = [10, 12, 20, 24, 18, 40]
demand = [20, 15, 25, 33, 21]

# 距离矩阵（公里）
dist_matrix = [
    [290, 115, 355, 715, 810],    # Verona
    [380, 340, 165, 380, 610],    # Perugia
    [505, 530, 285, 220, 450],    # Rome
    [655, 450, 155, 240, 315],    # Pescara
    [1010, 840, 550, 305, 95],    # Taranto
    [1072, 1097, 747, 372, 333]   # Lamezia
]

# 创建模型
model = gp.Model("ContainerTransport")

# 创建决策变量
x = {}
for i in range(len(warehouses)):
    for j in range(len(ports)):
        x[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{warehouses[i]}_{ports[j]}")

# 设置目标函数：最小化总成本
model.setObjective(
    gp.quicksum(30 * dist_matrix[i][j] * x[i, j] 
                for i in range(len(warehouses)) 
                for j in range(len(ports))),
    GRB.MINIMIZE
)

# 添加供应约束
for i in range(len(warehouses)):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(len(ports))) <= supply[i],
        name=f"supply_{warehouses[i]}"
    )

# 添加需求约束
for j in range(len(ports)):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(len(warehouses))) == demand[j],
        name=f"demand_{ports[j]}"
    )

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最优总成本: {model.ObjVal:.2f} 欧元\n")
    print("运输方案:")
    total_containers = 0
    for i in range(len(warehouses)):
        for j in range(len(ports)):
            if x[i, j].X > 0:
                print(f"  从 {warehouses[i]:8} 到 {ports[j]:8}: {x[i, j].X:2} 个集装箱")
                total_containers += x[i, j].X
    print(f"\n总运输集装箱数量: {total_containers}")
else:
    print("未找到最优解")
    print(f"求解状态: {model.status}")