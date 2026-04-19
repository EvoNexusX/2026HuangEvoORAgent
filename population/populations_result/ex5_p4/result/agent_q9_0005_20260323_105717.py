import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("Haus_Toys")

# 添加变量
x1 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x1")
x2 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x2")
x3 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x3")
x4 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x4")
w = model.addVar(vtype=GRB.BINARY, name="w")

# 设置目标函数
model.setObjective(5*x1 + 10*x2 + 8*x3 + 7*x4, GRB.MAXIMIZE)

# 添加约束
model.addConstr(12*x1 + 20*x2 + 15*x3 + 10*x4 <= 890, "wood")
model.addConstr(6*x1 + 3*x2 + 5*x3 + 4*x4 <= 500, "steel")
model.addConstr(x3 <= x4, "boat_train")
model.addConstr(x3 <= 59*x2, "boat_plane")
model.addConstr(x1 <= 74*(1 - w), "truck_excl")
model.addConstr(x4 <= 89*w, "train_excl")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最大利润: ${model.ObjVal:.2f}")
    print(f"玩具卡车生产数量: {x1.X:.0f}")
    print(f"玩具飞机生产数量: {x2.X:.0f}")
    print(f"玩具船生产数量: {x3.X:.0f}")
    print(f"玩具火车生产数量: {x4.X:.0f}")
else:
    print("未找到最优解")