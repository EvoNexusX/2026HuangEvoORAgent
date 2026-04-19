import gurobipy as gp
from gurobipy import GRB

# 数据准备
warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]

supply = [10, 12, 20, 24, 18, 40]
demand = [20, 15, 25, 33, 21]

distances = [
    [290, 115, 355, 715, 810],
    [380, 340, 165, 380, 610],
    [505, 530, 285, 220, 450],
    [655, 450, 155, 240, 315],
    [1010, 840, 550, 305, 95],
    [1072, 1097, 747, 372, 333]
]

cost_per_km = 30
truck_capacity = 2

# 创建模型
model = gp.Model("ContainerTransportation")

# 定义决策变量
x = model.addVars(len(warehouses), len(ports), vtype=GRB.INTEGER, name="x")
t = model.addVars(len(warehouses), len(ports), vtype=GRB.INTEGER, name="t")

# 设置目标函数
model.setObjective(
    gp.quicksum(
        cost_per_km * distances[i][j] * t[i, j]
        for i in range(len(warehouses))
        for j in range(len(ports))
    ),
    GRB.MINIMIZE
)

# 添加约束
# 1. 供应约束
for i in range(len(warehouses)):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(len(ports))) <= supply[i],
        name=f"Supply_{warehouses[i]}"
    )

# 2. 需求约束
for j in range(len(ports)):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(len(warehouses))) == demand[j],
        name=f"Demand_{ports[j]}"
    )

# 3. 卡车容量约束
for i in range(len(warehouses)):
    for j in range(len(ports)):
        model.addConstr(
            x[i, j] <= truck_capacity * t[i, j],
            name=f"Capacity_{warehouses[i]}_{ports[j]}"
        )

# 求解参数设置
model.Params.OutputFlag = 1
model.Params.TimeLimit = 300
model.Params.MIPGap = 0.01

# 求解模型
model.optimize()

# 结果输出
if model.status == GRB.OPTIMAL:
    print(f"最优总成本: {model.objVal:.2f} 欧元")
    print("\n运输方案详情:")
    for i in range(len(warehouses)):
        for j in range(len(ports)):
            if x[i, j].x > 0:
                transport_cost = cost_per_km * distances[i][j] * t[i, j].x
                print(f"{warehouses[i]} -> {ports[j]}: {x[i, j].x:.0f} 个容器, {t[i, j].x:.0f} 辆卡车, 成本: {transport_cost:.2f} 欧元")
    
    print("\n供应使用情况:")
    for i in range(len(warehouses)):
        used = sum(x[i, j].x for j in range(len(ports)))
        print(f"{warehouses[i]}: {used:.0f}/{supply[i]} 个容器")
    
    print("\n需求满足情况:")
    for j in range(len(ports)):
        received = sum(x[i, j].x for i in range(len(warehouses)))
        print(f"{ports[j]}: {received:.0f}/{demand[j]} 个容器")
    
    total_trucks = sum(t[i, j].x for i in range(len(warehouses)) for j in range(len(ports)))
    print(f"\n总卡车使用量: {total_trucks:.0f} 辆")
else:
    print("未找到最优解")
    print(f"求解状态: {model.status}")