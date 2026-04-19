import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("ContainerTransport")

# 集合定义
warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]

# 参数定义
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

distance = {
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
    ("Lamezia", "Bari"): 333
}

r = 30
cap = 2

# 决策变量
x = model.addVars(warehouses, ports, vtype=GRB.INTEGER, name="x")
y = model.addVars(warehouses, ports, vtype=GRB.INTEGER, name="y")

# 目标函数
model.setObjective(
    gp.quicksum(r * distance[i, j] * y[i, j] for i in warehouses for j in ports),
    GRB.MINIMIZE
)

# 约束条件
# 供应约束
supply_constr = model.addConstrs(
    (gp.quicksum(x[i, j] for j in ports) <= supply[i] for i in warehouses),
    name="supply"
)

# 需求约束
demand_constr = model.addConstrs(
    (gp.quicksum(x[i, j] for i in warehouses) == demand[j] for j in ports),
    name="demand"
)

# 卡车容量约束
capacity_constr = model.addConstrs(
    (x[i, j] <= cap * y[i, j] for i in warehouses for j in ports),
    name="capacity"
)

# 求解
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最小总成本: {model.ObjVal:.2f} 欧元")
    print("\n运输方案（集装箱数量）:")
    for i in warehouses:
        for j in ports:
            if x[i, j].x > 0:
                print(f"  {i} -> {j}: {int(x[i, j].x)} 个集装箱")
    print("\n卡车调度方案:")
    for i in warehouses:
        for j in ports:
            if y[i, j].x > 0:
                print(f"  {i} -> {j}: {int(y[i, j].x)} 辆卡车")
else:
    print(f"求解失败，状态码: {model.status}")