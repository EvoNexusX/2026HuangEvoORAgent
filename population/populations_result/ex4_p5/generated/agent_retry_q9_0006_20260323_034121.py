# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

# 1. 模型初始化
model = gp.Model("ToyProduction")
model.setParam('OutputFlag', 1)  # 显示求解过程

# 2. 定义决策变量
# 连续变量：生产数量
T = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="T")
A = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="A")
B = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="B")
R = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="R")

# 二进制变量：是否生产
y_T = model.addVar(vtype=GRB.BINARY, name="y_T")
y_A = model.addVar(vtype=GRB.BINARY, name="y_A")
y_B = model.addVar(vtype=GRB.BINARY, name="y_B")
y_R = model.addVar(vtype=GRB.BINARY, name="y_R")

# 3. 设置目标函数：最大化利润
model.setObjective(5 * T + 10 * A + 8 * B + 7 * R, GRB.MAXIMIZE)

# 4. 添加约束条件
# 木材约束
model.addConstr(12 * T + 20 * A + 15 * B + 10 * R <= 890, name="wood_constraint")

# 钢铁约束
model.addConstr(6 * T + 3 * A + 5 * B + 4 * R <= 500, name="steel_constraint")

# 船的数量不超过火车
model.addConstr(B <= R, name="B_le_R")

# 卡车和火车互斥
model.addConstr(y_T + y_R <= 1, name="exclusive_T_R")

# 船 implies 飞机
model.addConstr(y_B <= y_A, name="B_implies_A")

# 连接约束（生产数量与指示变量关联）
model.addConstr(T <= 75 * y_T, name="link_T")
model.addConstr(A <= 45 * y_A, name="link_A")
model.addConstr(B <= 60 * y_B, name="link_B")
model.addConstr(R <= 89 * y_R, name="link_R")

# 5. 求解模型
model.optimize()

# 6. 结果提取与输出
if model.status == GRB.OPTIMAL:
    print(f"Optimal objective value (max profit): ${model.objVal:.2f}")
    print("\nOptimal production plan:")
    print(f"Trucks (T): {T.x:.2f}")
    print(f"Airplanes (A): {A.x:.2f}")
    print(f"Boats (B): {B.x:.2f}")
    print(f"Trains (R): {R.x:.2f}")
    print(f"Produce trucks? (y_T): {int(y_T.x)}")
    print(f"Produce airplanes? (y_A): {int(y_A.x)}")
    print(f"Produce boats? (y_B): {int(y_B.x)}")
    print(f"Produce trains? (y_R): {int(y_R.x)}")
    
    # 计算资源使用情况
    wood_used = 12 * T.x + 20 * A.x + 15 * B.x + 10 * R.x
    steel_used = 6 * T.x + 3 * A.x + 5 * B.x + 4 * R.x
    print(f"\nResource usage:")
    print(f"Wood used: {wood_used:.2f} / 890")
    print(f"Steel used: {steel_used:.2f} / 500")
    
elif model.status == GRB.INFEASIBLE:
    print("Model is infeasible")
    # 计算不可行约束（IIS）以供分析
    model.computeIIS()
    model.write("model.ilp")
    print("IIS written to file 'model.ilp'")
    
elif model.status == GRB.UNBOUNDED:
    print("Model is unbounded")
    
else:
    print(f"Optimization terminated with status {model.status}")