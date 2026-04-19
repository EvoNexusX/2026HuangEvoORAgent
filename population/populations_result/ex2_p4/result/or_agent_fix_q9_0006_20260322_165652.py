# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("Toy_Manufacturing")

# 定义决策变量
T = model.addVar(vtype=GRB.INTEGER, lb=0, name="T")
A = model.addVar(vtype=GRB.INTEGER, lb=0, name="A")
B = model.addVar(vtype=GRB.INTEGER, lb=0, name="B")
R = model.addVar(vtype=GRB.INTEGER, lb=0, name="R")

y_T = model.addVar(vtype=GRB.BINARY, name="y_T")
y_A = model.addVar(vtype=GRB.BINARY, name="y_A")
y_B = model.addVar(vtype=GRB.BINARY, name="y_B")
y_R = model.addVar(vtype=GRB.BINARY, name="y_R")

# 设置目标函数
model.setObjective(5*T + 10*A + 8*B + 7*R, GRB.MAXIMIZE)

# 添加约束
# 木材约束
model.addConstr(12*T + 20*A + 15*B + 10*R <= 890, "Wood")
# 钢材约束
model.addConstr(6*T + 3*A + 5*B + 4*R <= 500, "Steel")
# 逻辑约束：生产卡车则不生产火车
model.addConstr(y_T + y_R <= 1, "Truck_Train_Exclusive")
# 逻辑约束：生产船则生产飞机
model.addConstr(y_B <= y_A, "Boat_Implies_Airplane")
# 数量约束：船的数量不超过火车数量
model.addConstr(B <= R, "Boat_leq_Train")
# 生产指示约束
model.addConstr(T <= 74 * y_T, "Ind_T")
model.addConstr(A <= 44 * y_A, "Ind_A")
model.addConstr(B <= 59 * y_B, "Ind_B")
model.addConstr(R <= 89 * y_R, "Ind_R")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最优目标值: ${model.ObjVal:.2f}")
    print("生产数量:")
    print(f"  卡车: {int(T.X)}")
    print(f"  飞机: {int(A.X)}")
    print(f"  船: {int(B.X)}")
    print(f"  火车: {int(R.X)}")
    print("是否生产:")
    print(f"  卡车: {int(y_T.X)}")
    print(f"  飞机: {int(y_A.X)}")
    print(f"  船: {int(y_B.X)}")
    print(f"  火车: {int(y_R.X)}")
else:
    print("未找到最优解")