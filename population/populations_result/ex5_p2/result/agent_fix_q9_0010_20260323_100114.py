# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

# 创建Gurobi环境和模型
env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)  # 关闭求解器输出
env.start()
model = gp.Model("ProductionPlanning", env=env)

# 决策变量定义
# 整数变量：x_T, x_A, x_B, x_R
x_T = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_T")
x_A = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_A")
x_B = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_B")
x_R = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_R")

# 二元变量：y_T, y_A, y_B, y_R
y_T = model.addVar(vtype=GRB.BINARY, name="y_T")
y_A = model.addVar(vtype=GRB.BINARY, name="y_A")
y_B = model.addVar(vtype=GRB.BINARY, name="y_B")
y_R = model.addVar(vtype=GRB.BINARY, name="y_R")

# 设置目标函数
model.setObjective(5*x_T + 10*x_A + 8*x_B + 7*x_R, GRB.MAXIMIZE)

# 添加约束条件
# 1. 木材资源约束
model.addConstr(12*x_T + 20*x_A + 15*x_B + 10*x_R <= 890, "wood_constraint")

# 2. 钢铁资源约束
model.addConstr(6*x_T + 3*x_A + 5*x_B + 4*x_R <= 500, "steel_constraint")

# 3. 生产指示约束 (Big-M法)
# 各产品的上限值
M_T = 74
M_A = 44
M_B = 59
M_R = 89

# x_i <= M_i * y_i
model.addConstr(x_T <= M_T * y_T, "bigM_T_upper")
model.addConstr(x_A <= M_A * y_A, "bigM_A_upper")
model.addConstr(x_B <= M_B * y_B, "bigM_B_upper")
model.addConstr(x_R <= M_R * y_R, "bigM_R_upper")

# x_i >= y_i (保证如果y_i=1，则x_i至少为1)
model.addConstr(x_T >= y_T, "bigM_T_lower")
model.addConstr(x_A >= y_A, "bigM_A_lower")
model.addConstr(x_B >= y_B, "bigM_B_lower")
model.addConstr(x_R >= y_R, "bigM_R_lower")

# 4. 卡车与火车互斥逻辑约束
model.addConstr(y_T + y_R <= 1, "truck_train_exclusive")

# 5. 船与飞机逻辑约束
model.addConstr(y_B <= y_A, "boat_requires_airplane")

# 6. 船与火车数量关系约束
model.addConstr(x_B <= x_R, "boat_leq_train")

# 求解模型
model.optimize()

# 检查求解状态并输出结果
if model.status == GRB.OPTIMAL:
    print(f"Optimal profit: ${model.objVal:.2f}")
    print("\nOptimal production plan:")
    print(f"  Number of trucks (x_T): {x_T.x:.0f}")
    print(f"  Number of airplanes (x_A): {x_A.x:.0f}")
    print(f"  Number of boats (x_B): {x_B.x:.0f}")
    print(f"  Number of trains (x_R): {x_R.x:.0f}")
    print(f"\nProduction indicators:")
    print(f"  y_T: {y_T.x:.0f}")
    print(f"  y_A: {y_A.x:.0f}")
    print(f"  y_B: {y_B.x:.0f}")
    print(f"  y_R: {y_R.x:.0f}")
else:
    print(f"Optimal solution not found. Status: {model.status}")

# 释放资源
model.dispose()
env.dispose()