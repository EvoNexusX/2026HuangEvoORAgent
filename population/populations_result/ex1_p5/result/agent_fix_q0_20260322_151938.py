# -*- coding: utf-8 -*-
from gurobipy import *

# 创建模型
model = Model("simpleLP")

# 定义变量
x = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x")
y = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y")

# 更新模型以集成变量
model.update()

# 设置目标函数
model.setObjective(x + y, GRB.MINIMIZE)

# 添加约束
model.addConstr(x + y >= 5, "c0")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print('Optimal objective value:', model.objVal)
    print('x:', x.x)
    print('y:', y.x)
else:
    print('No optimal solution found. Status:', model.status)