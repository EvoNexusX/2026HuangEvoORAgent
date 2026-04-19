import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("HausToys")

# 决策变量
x1 = model.addVar(vtype=GRB.INTEGER, name="x1")  # 卡车数量
x2 = model.addVar(vtype=GRB.INTEGER, name="x2")  # 飞机数量
x3 = model.addVar(vtype=GRB.INTEGER, name="x3")  # 船数量
x4 = model.addVar(vtype=GRB.INTEGER, name="x4")  # 火车数量
y = model.addVar(vtype=GRB.BINARY, name="y")     # 卡车-火车互斥指示变量
z = model.addVar(vtype=GRB.BINARY, name="z")     # 船生产指示变量

# 设置目标函数
model.setObjective(5*x1 + 10*x2 + 8*x3 + 7*x4, GRB.MAXIMIZE)

# 添加约束条件
# 资源约束
model.addConstr(12*x1 + 20*x2 + 15*x3 + 10*x4 <= 890, "Wood")
model.addConstr(6*x1 + 3*x2 + 5*x3 + 4*x4 <= 500, "Steel")

# 卡车与火车互斥约束
M1 = 74
M2 = 89
model.addConstr(x1 <= M1 * y, "Truck_activation")
model.addConstr(x4 <= M2 * (1 - y), "Train_activation")

# 船与飞机的逻辑约束
M3 = 59
model.addConstr(x3 <= M3 * z, "Boat_activation")
model.addConstr(x2 >= z, "Airplane_if_boat")

# 船与火车的数量关系
model.addConstr(x3 <= x4, "Boat_vs_Train")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    print(f"Number of trucks (x1): {x1.X}")
    print(f"Number of airplanes (x2): {x2.X}")
    print(f"Number of boats (x3): {x3.X}")
    print(f"Number of trains (x4): {x4.X}")
    print(f"Indicator y (1 if trucks are made): {y.X}")
    print(f"Indicator z (1 if boats are made): {z.X}")
    print(f"Maximum profit: ${model.ObjVal}")
else:
    print("No optimal solution found. Status:", model.status)