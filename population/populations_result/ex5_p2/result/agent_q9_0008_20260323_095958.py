import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("ProductionPlanning")

# 定义决策变量
x_T = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_T")  # 卡车数量
x_A = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_A")  # 飞机数量
x_B = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_B")  # 船数量
x_R = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_R")  # 火车数量

y_T = model.addVar(vtype=GRB.BINARY, name="y_T")  # 卡车生产指示变量
y_A = model.addVar(vtype=GRB.BINARY, name="y_A")  # 飞机生产指示变量
y_B = model.addVar(vtype=GRB.BINARY, name="y_B")  # 船生产指示变量
y_R = model.addVar(vtype=GRB.BINARY, name="y_R")  # 火车生产指示变量

# 设置目标函数：最大化利润
model.setObjective(5 * x_T + 10 * x_A + 8 * x_B + 7 * x_R, GRB.MAXIMIZE)

# 添加约束条件
# 1. 木材资源约束
model.addConstr(12 * x_T + 20 * x_A + 15 * x_B + 10 * x_R <= 890, "wood_constraint")

# 2. 钢铁资源约束
model.addConstr(6 * x_T + 3 * x_A + 5 * x_B + 4 * x_R <= 500, "steel_constraint")

# 3. 生产指示约束 (Big-M法)
M_T = 74  # 基于资源约束计算的卡车上限
M_A = 44  # 基于资源约束计算的飞机上限
M_B = 59  # 基于资源约束计算的船上限
M_R = 89  # 基于资源约束计算的火车上限

model.addConstr(x_T <= M_T * y_T, "indicator_T_upper")
model.addConstr(x_T >= y_T, "indicator_T_lower")

model.addConstr(x_A <= M_A * y_A, "indicator_A_upper")
model.addConstr(x_A >= y_A, "indicator_A_lower")

model.addConstr(x_B <= M_B * y_B, "indicator_B_upper")
model.addConstr(x_B >= y_B, "indicator_B_lower")

model.addConstr(x_R <= M_R * y_R, "indicator_R_upper")
model.addConstr(x_R >= y_R, "indicator_R_lower")

# 4. 卡车与火车互斥逻辑约束
model.addConstr(y_T + y_R <= 1, "truck_train_mutex")

# 5. 船与飞机逻辑约束（若生产船则必须生产飞机）
model.addConstr(y_B <= y_A, "boat_plane_logic")

# 6. 船与火车数量关系约束
model.addConstr(x_B <= x_R, "boat_train_quantity")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最优目标值: {model.objVal:.2f}")
    print("最优解:")
    print(f"  卡车数量 x_T = {x_T.x:.0f}")
    print(f"  飞机数量 x_A = {x_A.x:.0f}")
    print(f"  船数量   x_B = {x_B.x:.0f}")
    print(f"  火车数量 x_R = {x_R.x:.0f}")
    print(f"  卡车生产指示 y_T = {y_T.x:.0f}")
    print(f"  飞机生产指示 y_A = {y_A.x:.0f}")
    print(f"  船生产指示   y_B = {y_B.x:.0f}")
    print(f"  火车生产指示 y_R = {y_R.x:.0f}")
else:
    print(f"求解未达到最优，状态码: {model.status}")