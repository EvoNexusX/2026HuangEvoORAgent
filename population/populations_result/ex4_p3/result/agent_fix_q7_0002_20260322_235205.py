# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

# 仓库和港口信息
warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]

# 供应量和需求量
supply = [10, 12, 20, 24, 18, 40]
demand = [20, 15, 25, 33, 21]

# 距离矩阵 (公里)
distances = [
    [290, 115, 355, 715, 810],   # Verona
    [380, 340, 165, 380, 610],   # Perugia
    [505, 530, 285, 220, 450],   # Rome
    [655, 450, 155, 240, 315],   # Pescara
    [1010, 840, 550, 305, 95],   # Taranto
    [1072, 1097, 747, 372, 333]  # Lamezia
]

# 运输成本率 (欧元/公里)
cost_per_km = 30

# 卡车容量
truck_capacity = 2

# 创建模型
model = gp.Model("EmptyContainerTransportation")

# 创建变量
x = {}  # 从仓库i运往港口j的集装箱数量
y = {}  # 从仓库i到港口j的卡车数量

for i in range(len(warehouses)):
    for j in range(len(ports)):
        x[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{i}_{j}")
        y[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"y_{i}_{j}")

# 设置目标函数：最小化总运输成本
model.setObjective(
    gp.quicksum(
        cost_per_km * distances[i][j] * y[i, j]
        for i in range(len(warehouses))
        for j in range(len(ports))
    ),
    GRB.MINIMIZE
)

# 添加约束

# 供应约束：每个仓库运出的集装箱总数不超过其库存
for i in range(len(warehouses)):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(len(ports))) <= supply[i],
        name=f"Supply_{warehouses[i]}"
    )

# 需求约束：每个港口收到的集装箱总数等于其需求
for j in range(len(ports)):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(len(warehouses))) == demand[j],
        name=f"Demand_{ports[j]}"
    )

# 卡车容量约束：每辆卡车最多装载2个集装箱
for i in range(len(warehouses)):
    for j in range(len(ports)):
        model.addConstr(
            x[i, j] <= truck_capacity * y[i, j],
            name=f"TruckCapacity_{i}_{j}"
        )

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最小运输成本: {model.ObjVal:.2f} 欧元")
    print("\n集装箱运输方案:")
    print("仓库 -> 港口 : 集装箱数量 (卡车数量)")
    
    for i in range(len(warehouses)):
        for j in range(len(ports)):
            if x[i, j].X > 0:
                print(f"{warehouses[i]} -> {ports[j]} : {int(x[i, j].X)} ({int(y[i, j].X)})")
    
    print("\n仓库库存使用情况:")
    for i in range(len(warehouses)):
        total_sent = sum(x[i, j].X for j in range(len(ports)))
        print(f"{warehouses[i]}: 运出 {int(total_sent)} / 库存 {supply[i]}")
    
    print("\n港口需求满足情况:")
    for j in range(len(ports)):
        total_received = sum(x[i, j].X for i in range(len(warehouses)))
        print(f"{ports[j]}: 收到 {int(total_received)} / 需求 {demand[j]}")
else:
    print("未找到最优解")
    print(f"求解状态: {model.status}")