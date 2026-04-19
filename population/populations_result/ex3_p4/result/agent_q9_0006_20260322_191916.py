import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("Toy_Production")

# 定义决策变量
x1 = model.addVar(lb=0, vtype=GRB.INTEGER, name="x1")  # 卡车数量
x2 = model.addVar(lb=0, vtype=GRB.INTEGER, name="x2")  # 飞机数量
x3 = model.addVar(lb=0, vtype=GRB.INTEGER, name="x3")  # 船数量
x4 = model.addVar(lb=0, vtype=GRB.INTEGER, name="x4")  # 火车数量

b1 = model.addVar(vtype=GRB.BINARY, name="b1")  # 是否生产卡车
b2 = model.addVar(vtype=GRB.BINARY, name="b2")  # 是否生产飞机
b3 = model.addVar(vtype=GRB.BINARY, name="b3")  # 是否生产船
b4 = model.addVar(vtype=GRB.BINARY, name="b4")  # 是否生产火车

# 设置目标函数
model.setObjective(5*x1 + 10*x2 + 8*x3 + 7*x4, GRB.MAXIMIZE)

# 添加约束

# 资源约束
model.addConstr(12*x1 + 20*x2 + 15*x3 + 10*x4 <= 890, "Wood")
model.addConstr(6*x1 + 3*x2 + 5*x3 + 4*x4 <= 500, "Steel")

# 数量与二元变量关联约束（大M法）
M1 = 74
M2 = 44
M3 = 59
M4 = 89

model.addConstr(x1 <= M1 * b1, "M1_upper")
model.addConstr(x1 >= b1, "M1_lower")

model.addConstr(x2 <= M2 * b2, "M2_upper")
model.addConstr(x2 >= b2, "M2_lower")

model.addConstr(x3 <= M3 * b3, "M3_upper")
model.addConstr(x3 >= b3, "M3_lower")

model.addConstr(x4 <= M4 * b4, "M4_upper")
model.addConstr(x4 >= b4, "M4_lower")

# 逻辑约束
model.addConstr(b4 <= 1 - b1, "No_train_if_truck")
model.addConstr(b2 >= b3, "If_boat_then_airplane")
model.addConstr(x3 <= x4, "Boat_no_more_than_train")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最优目标值: {model.objVal}")
    print(f"卡车生产数量: {x1.x}")
    print(f"飞机生产数量: {x2.x}")
    print(f"船生产数量: {x3.x}")
    print(f"火车生产数量: {x4.x}")
    print(f"是否生产卡车 (b1): {b1.x}")
    print(f"是否生产飞机 (b2): {b2.x}")
    print(f"是否生产船 (b3): {b3.x}")
    print(f"是否生产火车 (b4): {b4.x}")
else:
    print("未找到最优解")