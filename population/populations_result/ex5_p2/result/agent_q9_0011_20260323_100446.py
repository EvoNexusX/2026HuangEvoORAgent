import gurobipy as gp
from gurobipy import GRB

# 创建环境并设置参数
env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()
model = gp.Model("HausToys_Production", env=env)

# 变量定义
x_T = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_T")
x_A = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_A")
x_B = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_B")
x_R = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_R")

y_T = model.addVar(vtype=GRB.BINARY, name="y_T")
y_A = model.addVar(vtype=GRB.BINARY, name="y_A")
y_B = model.addVar(vtype=GRB.BINARY, name="y_B")
y_R = model.addVar(vtype=GRB.BINARY, name="y_R")

# 目标函数
model.setObjective(5*x_T + 10*x_A + 8*x_B + 7*x_R, GRB.MAXIMIZE)

# 资源约束
model.addConstr(12*x_T + 20*x_A + 15*x_B + 10*x_R <= 890, "wood")
model.addConstr(6*x_T + 3*x_A + 5*x_B + 4*x_R <= 500, "steel")

# 逻辑约束
model.addConstr(y_T + y_R <= 1, "truck_train_excl")
model.addConstr(y_B <= y_A, "boat_implies_airplane")
model.addConstr(x_B <= x_R, "boat_leq_train")

# 修改：移除有问题的生产指示约束，改用更精确的Big-M方法
# 计算更精确的上界
M_T = 74  # min(890/12, 500/6) 取整
M_A = 44  # min(890/20, 500/3) 取整
M_B = 59  # min(890/15, 500/5) 取整
M_R = 89  # min(890/10, 500/4) 取整

# 修正后的Big-M约束：只有当y_i=1时，x_i才可能大于0
model.addConstr(x_T <= M_T * y_T, "bigM_T_upper")
model.addConstr(x_A <= M_A * y_A, "bigM_A_upper")
model.addConstr(x_B <= M_B * y_B, "bigM_B_upper")
model.addConstr(x_R <= M_R * y_R, "bigM_R_upper")

# 修改：使用更合理的下界约束
# 只有当y_i=1时，x_i才必须≥1
model.addConstr(x_T >= y_T, "bigM_T_lower")
model.addConstr(x_A >= y_A, "bigM_A_lower")
model.addConstr(x_B >= y_B, "bigM_B_lower")
model.addConstr(x_R >= y_R, "bigM_R_lower")

# 求解
model.optimize()

# 结果输出
if model.status == GRB.OPTIMAL:
    print(f"Optimal Profit: ${model.objVal:.2f}")
    print(f"Trucks: {x_T.x:.0f}")
    print(f"Airplanes: {x_A.x:.0f}")
    print(f"Boats: {x_B.x:.0f}")
    print(f"Trains: {x_R.x:.0f}")
elif model.status == GRB.INFEASIBLE:
    print("Model is infeasible")
    # 计算IIS以诊断不可行性
    model.computeIIS()
    model.write("model.ilp")
    print("IIS written to model.ilp")
else:
    print(f"Optimization ended with status: {model.status}")

# 释放资源
model.dispose()
env.dispose()