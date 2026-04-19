import gurobipy as gp
from gurobipy import GRB

# 步骤2：初始化模型
model = gp.Model("Empty_Container_Transportation")

# 步骤3：定义集合与参数
warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]

supply = {
    "Verona": 10,
    "Perugia": 12,
    "Rome": 20,
    "Pescara": 24,
    "Taranto": 18,
    "Lamezia": 40
}

demand = {
    "Genoa": 20,
    "Venice": 15,
    "Ancona": 25,
    "Naples": 33,
    "Bari": 21
}

distances = {
    ("Verona", "Genoa"): 290,
    ("Verona", "Venice"): 115,
    ("Verona", "Ancona"): 355,
    ("Verona", "Naples"): 715,
    ("Verona", "Bari"): 810,
    ("Perugia", "Genoa"): 380,
    ("Perugia", "Venice"): 340,
    ("Perugia", "Ancona"): 165,
    ("Perugia", "Naples"): 380,
    ("Perugia", "Bari"): 610,
    ("Rome", "Genoa"): 505,
    ("Rome", "Venice"): 530,
    ("Rome", "Ancona"): 285,
    ("Rome", "Naples"): 220,
    ("Rome", "Bari"): 450,
    ("Pescara", "Genoa"): 655,
    ("Pescara", "Venice"): 450,
    ("Pescara", "Ancona"): 155,
    ("Pescara", "Naples"): 240,
    ("Pescara", "Bari"): 315,
    ("Taranto", "Genoa"): 1010,
    ("Taranto", "Venice"): 840,
    ("Taranto", "Ancona"): 550,
    ("Taranto", "Naples"): 305,
    ("Taranto", "Bari"): 95,
    ("Lamezia", "Genoa"): 1072,
    ("Lamezia", "Venice"): 1097,
    ("Lamezia", "Ancona"): 747,
    ("Lamezia", "Naples"): 372,
    ("Lamezia", "Bari"): 333,
}

c = 30  # 每公里成本率（欧元/公里）

# 步骤4：添加决策变量
x = {}
y = {}

for i in warehouses:
    for j in ports:
        x[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{i}_{j}")
        y[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"y_{i}_{j}")

# 步骤5：设置目标函数
model.setObjective(
    gp.quicksum(c * distances[i, j] * y[i, j] for i in warehouses for j in ports),
    GRB.MINIMIZE
)

# 步骤6：添加约束条件
# 供应约束
for i in warehouses:
    model.addConstr(
        gp.quicksum(x[i, j] for j in ports) <= supply[i],
        name=f"Supply_{i}"
    )

# 需求约束
for j in ports:
    model.addConstr(
        gp.quicksum(x[i, j] for i in warehouses) == demand[j],
        name=f"Demand_{j}"
    )

# 卡车容量约束和装载下限约束
for i in warehouses:
    for j in ports:
        model.addConstr(x[i, j] <= 2 * y[i, j], name=f"Capacity_{i}_{j}")
        model.addConstr(x[i, j] >= y[i, j], name=f"MinLoad_{i}_{j}")

# 步骤7：配置求解参数（可选）
model.setParam('OutputFlag', 1)  # 显示求解过程

# 步骤8：求解模型
model.optimize()

# 步骤9：提取与分析结果
if model.status == GRB.OPTIMAL:
    print(f"\n最优总运输成本: {model.ObjVal:.2f} 欧元")
    
    # 输出非零运输量
    print("\n非零运输方案：")
    total_containers = 0
    for i in warehouses:
        for j in ports:
            if x[i, j].X > 0:
                containers = x[i, j].X
                trucks = y[i, j].X
                cost_per_route = c * distances[i, j] * trucks
                total_containers += containers
                print(f"  从 {i} 到 {j}: {containers} 个集装箱, {trucks} 辆卡车, 路线成本: {cost_per_route:.2f} 欧元")
    
    print(f"\n总运输集装箱数量: {total_containers}")
    print(f"总需求: {sum(demand.values())}")
    print(f"总供应: {sum(supply.values())}")
    
    # 验证需求满足情况
    for j in ports:
        received = sum(x[i, j].X for i in warehouses)
        print(f"  港口 {j}: 需求 {demand[j]}, 收到 {received}, {'满足' if received == demand[j] else '不满足'}")
    
    # 检查卡车约束
    print("\n卡车使用验证：")
    for i in warehouses:
        for j in ports:
            if x[i, j].X > 0:
                containers = x[i, j].X
                trucks = y[i, j].X
                if not (containers >= trucks and containers <= 2 * trucks):
                    print(f"  警告: 路线 {i}-{j} 违反卡车约束")
                else:
                    print(f"  路线 {i}-{j}: {containers} 个集装箱, {trucks} 辆卡车 (满足 {trucks} <= {containers} <= {2*trucks})")
else:
    print("未找到最优解")
    print(f"求解状态: {model.status}")