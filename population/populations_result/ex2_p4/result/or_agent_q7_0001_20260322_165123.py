import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("ContainerTransport")

# 数据定义
warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]

supply = [10, 12, 20, 24, 18, 40]
demand = [20, 15, 25, 33, 21]

# 距离矩阵（公里）
distances = [
    [290, 115, 355, 715, 810],
    [380, 340, 165, 380, 610],
    [505, 530, 285, 220, 450],
    [655, 450, 155, 240, 315],
    [1010, 840, 550, 305, 95],
    [1072, 1097, 747, 372, 333]
]

# 成本矩阵（欧元/集装箱）
cost_per_km = 30
costs = [[dist * cost_per_km for dist in row] for row in distances]

# 创建变量
x = model.addVars(len(warehouses), len(ports), lb=0, vtype=GRB.CONTINUOUS, name="x")

# 设置目标函数
model.setObjective(
    gp.quicksum(
        costs[i][j] * x[i, j]
        for i in range(len(warehouses))
        for j in range(len(ports))
    ),
    GRB.MINIMIZE
)

# 添加供应约束
for i in range(len(warehouses)):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(len(ports))) <= supply[i],
        name=f"Supply_{warehouses[i]}"
    )

# 添加需求约束
for j in range(len(ports)):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(len(warehouses))) == demand[j],
        name=f"Demand_{ports[j]}"
    )

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最优总成本: {model.objVal:.2f} 欧元")
    print("\n最优运输方案:")
    for i in range(len(warehouses)):
        for j in range(len(ports)):
            if x[i, j].x > 1e-6:
                print(f"{warehouses[i]} -> {ports[j]}: {x[i, j].x:.0f} 个集装箱")
    
    # 计算并输出卡车数量
    total_containers = sum(demand)
    trucks_needed = (total_containers + 1) // 2  # 向上取整
    print(f"\n需要卡车数量: {trucks_needed} 辆（每辆最多载2个集装箱）")
else:
    print("未找到最优解")