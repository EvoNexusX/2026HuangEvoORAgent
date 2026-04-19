import gurobipy as gp
from gurobipy import GRB

# 步骤1：数据定义
warehouses = ['Verona', 'Perugia', 'Rome', 'Pescara', 'Taranto', 'Lamezia']
ports = ['Genoa', 'Venice', 'Ancona', 'Naples', 'Bari']

supply = {
    'Verona': 10,
    'Perugia': 12,
    'Rome': 20,
    'Pescara': 24,
    'Taranto': 18,
    'Lamezia': 40
}

demand = {
    'Genoa': 20,
    'Venice': 15,
    'Ancona': 25,
    'Naples': 33,
    'Bari': 21
}

# 距离矩阵（公里）
distance = {
    ('Verona', 'Genoa'): 290,
    ('Verona', 'Venice'): 115,
    ('Verona', 'Ancona'): 355,
    ('Verona', 'Naples'): 715,
    ('Verona', 'Bari'): 810,
    ('Perugia', 'Genoa'): 380,
    ('Perugia', 'Venice'): 340,
    ('Perugia', 'Ancona'): 165,
    ('Perugia', 'Naples'): 380,
    ('Perugia', 'Bari'): 610,
    ('Rome', 'Genoa'): 505,
    ('Rome', 'Venice'): 530,
    ('Rome', 'Ancona'): 285,
    ('Rome', 'Naples'): 220,
    ('Rome', 'Bari'): 450,
    ('Pescara', 'Genoa'): 655,
    ('Pescara', 'Venice'): 450,
    ('Pescara', 'Ancona'): 155,
    ('Pescara', 'Naples'): 240,
    ('Pescara', 'Bari'): 315,
    ('Taranto', 'Genoa'): 1010,
    ('Taranto', 'Venice'): 840,
    ('Taranto', 'Ancona'): 550,
    ('Taranto', 'Naples'): 305,
    ('Taranto', 'Bari'): 95,
    ('Lamezia', 'Genoa'): 1072,
    ('Lamezia', 'Venice'): 1097,
    ('Lamezia', 'Ancona'): 747,
    ('Lamezia', 'Naples'): 372,
    ('Lamezia', 'Bari'): 333
}

# 成本矩阵（欧元）：距离 * 30
cost = {key: distance[key] * 30 for key in distance}

# 步骤2：创建模型
model = gp.Model("ContainerTransport")

# 步骤3：添加决策变量
x = model.addVars(warehouses, ports, vtype=GRB.INTEGER, lb=0, name="x")

# 步骤4：设置目标函数
model.setObjective(
    gp.quicksum(cost[i, j] * x[i, j] for i in warehouses for j in ports),
    GRB.MINIMIZE
)

# 步骤5：添加供应约束
model.addConstrs(
    (x.sum(i, '*') <= supply[i] for i in warehouses),
    name="supply"
)

# 步骤6：添加需求约束
model.addConstrs(
    (x.sum('*', j) == demand[j] for j in ports),
    name="demand"
)

# 步骤7：求解模型
model.optimize()

# 步骤8：输出结果
if model.status == GRB.OPTIMAL:
    print(f"最优总成本: {model.ObjVal:.2f} 欧元")
    print("\n运输方案:")
    for i in warehouses:
        for j in ports:
            flow = x[i, j].X
            if flow > 0:
                print(f"  从 {i:8} 到 {j:8}: {flow:2} 个集装箱 (成本: {cost[i, j] * flow:.2f} 欧元)")
else:
    print("未找到最优解")