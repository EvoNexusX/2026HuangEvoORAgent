import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("EmptyContainerTransport")

# 数据定义
warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]

supply = [10, 12, 20, 24, 18, 40]
demand = [20, 15, 25, 33, 21]

# 距离矩阵 (km)
dist_matrix = [
    [290, 115, 355, 715, 810],   # Verona
    [380, 340, 165, 380, 610],   # Perugia
    [505, 530, 285, 220, 450],   # Rome
    [655, 450, 155, 240, 315],   # Pescara
    [1010, 840, 550, 305, 95],   # Taranto
    [1072, 1097, 747, 372, 333]  # Lamezia
]

# 成本参数
rate_per_km = 30

# 创建决策变量
x = model.addVars(
    range(len(warehouses)), 
    range(len(ports)), 
    lb=0, 
    vtype=GRB.CONTINUOUS, 
    name="x"
)

# 设置目标函数
model.setObjective(
    gp.quicksum(
        rate_per_km * dist_matrix[i][j] * x[i, j] 
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
    print("\n运输方案 (集装箱数量):")
    print("从\\到", end="")
    for port in ports:
        print(f"\t{port}", end="")
    print()
    
    for i in range(len(warehouses)):
        print(f"{warehouses[i]}", end="")
        for j in range(len(ports)):
            val = x[i, j].X
            if val > 1e-6:
                print(f"\t{val:.0f}", end="")
            else:
                print("\t0", end="")
        print()
    
    # 验证约束
    print("\n仓库利用率:")
    for i in range(len(warehouses)):
        total_ship = sum(x[i, j].X for j in range(len(ports)))
        print(f"{warehouses[i]}: 运出 {total_ship:.0f}/{supply[i]} 集装箱")
    
    print("\n港口需求满足情况:")
    for j in range(len(ports)):
        total_receive = sum(x[i, j].X for i in range(len(warehouses)))
        print(f"{ports[j]}: 收到 {total_receive:.0f}/{demand[j]} 集装箱")
else:
    print(f"求解失败，状态码: {model.status}")