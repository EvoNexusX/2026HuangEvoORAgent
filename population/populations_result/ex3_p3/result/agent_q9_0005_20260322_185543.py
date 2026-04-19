import gurobipy as gp
from gurobipy import GRB

model = gp.Model("Toy_Production")

# 定义变量
x = model.addVars(['T', 'A', 'B', 'R'], lb=0, vtype=GRB.INTEGER, name="x")
y = model.addVars(['T', 'A', 'B', 'R'], vtype=GRB.BINARY, name="y")

# 设置目标函数
model.setObjective(5*x['T'] + 10*x['A'] + 8*x['B'] + 7*x['R'], GRB.MAXIMIZE)

# 添加约束
# 木材约束
model.addConstr(12*x['T'] + 20*x['A'] + 15*x['B'] + 10*x['R'] <= 890, "Wood")
# 钢材约束
model.addConstr(6*x['T'] + 3*x['A'] + 5*x['B'] + 4*x['R'] <= 500, "Steel")
# 船的数量不超过火车的数量
model.addConstr(x['B'] <= x['R'], "B_leq_R")
# 如果生产卡车则不生产火车
model.addConstr(y['T'] + y['R'] <= 1, "T_R_exclusive")
# 如果生产船则必须生产飞机
model.addConstr(y['B'] <= y['A'], "B_implies_A")

# 计算每个变量的上界
M_T = min(890 // 12, 500 // 6) if (890 // 12 > 0 and 500 // 6 > 0) else 0
M_A = min(890 // 20, 500 // 3) if (890 // 20 > 0 and 500 // 3 > 0) else 0
M_B = min(890 // 15, 500 // 5) if (890 // 15 > 0 and 500 // 5 > 0) else 0
M_R = min(890 // 10, 500 // 4) if (890 // 10 > 0 and 500 // 4 > 0) else 0

# 关联约束
model.addConstr(x['T'] <= M_T * y['T'], "link_T_upper")
model.addConstr(x['T'] >= y['T'], "link_T_lower")
model.addConstr(x['A'] <= M_A * y['A'], "link_A_upper")
model.addConstr(x['A'] >= y['A'], "link_A_lower")
model.addConstr(x['B'] <= M_B * y['B'], "link_B_upper")
model.addConstr(x['B'] >= y['B'], "link_B_lower")
model.addConstr(x['R'] <= M_R * y['R'], "link_R_upper")
model.addConstr(x['R'] >= y['R'], "link_R_lower")

# 求解
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最大利润: ${model.objVal:.2f}")
    print("生产数量:")
    for v in model.getVars():
        if v.varName.startswith('x'):
            print(f"  {v.varName}: {int(v.x)}")
else:
    print("未找到最优解")