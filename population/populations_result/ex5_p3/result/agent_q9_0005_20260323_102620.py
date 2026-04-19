import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("Haus_Toys_Production")

# 定义决策变量
x1 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x1")  # 卡车数量
x2 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x2")  # 飞机数量
x3 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x3")  # 船数量
x4 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x4")  # 火车数量

y1 = model.addVar(vtype=GRB.BINARY, name="y1")  # 是否生产卡车
y2 = model.addVar(vtype=GRB.BINARY, name="y2")  # 是否生产飞机
y3 = model.addVar(vtype=GRB.BINARY, name="y3")  # 是否生产船
y4 = model.addVar(vtype=GRB.BINARY, name="y4")  # 是否生产火车

# 更新模型以添加变量
model.update()

# 设置目标函数：最大化利润
model.setObjective(5*x1 + 10*x2 + 8*x3 + 7*x4, GRB.MAXIMIZE)

# 添加资源约束
# 木材约束：12*x1 + 20*x2 + 15*x3 + 10*x4 <= 890
model.addConstr(12*x1 + 20*x2 + 15*x3 + 10*x4 <= 890, "Wood_Constraint")

# 钢铁约束：6*x1 + 3*x2 + 5*x3 + 4*x4 <= 500
model.addConstr(6*x1 + 3*x2 + 5*x3 + 4*x4 <= 500, "Steel_Constraint")

# 大M约束：产量与二元变量关联
# 大M值：M1=74, M2=44, M3=59, M4=89
M1 = 74
M2 = 44
M3 = 59
M4 = 89

model.addConstr(x1 <= M1 * y1, "BigM_x1_upper")
model.addConstr(x1 >= y1, "BigM_x1_lower")
model.addConstr(x2 <= M2 * y2, "BigM_x2_upper")
model.addConstr(x2 >= y2, "BigM_x2_lower")
model.addConstr(x3 <= M3 * y3, "BigM_x3_upper")
model.addConstr(x3 >= y3, "BigM_x3_lower")
model.addConstr(x4 <= M4 * y4, "BigM_x4_upper")
model.addConstr(x4 >= y4, "BigM_x4_lower")

# 添加逻辑约束
# 卡车和火车互斥：y1 + y4 <= 1
model.addConstr(y1 + y4 <= 1, "Truck_Train_Exclusive")

# 如果生产船则必须生产飞机：y3 <= y2
model.addConstr(y3 <= y2, "Boat_requires_Airplane")

# 船的数量不超过火车的数量：x3 <= x4
model.addConstr(x3 <= x4, "Boat_leq_Train_Quantity")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print("最优解找到！")
    print(f"最大利润: ${model.objVal:.2f}")
    print("\n最优生产计划:")
    print(f"  卡车 (x1): {x1.x} 单位")
    print(f"  飞机 (x2): {x2.x} 单位")
    print(f"  船 (x3): {x3.x} 单位")
    print(f"  火车 (x4): {x4.x} 单位")
    print(f"  是否生产卡车 (y1): {int(y1.x)}")
    print(f"  是否生产飞机 (y2): {int(y2.x)}")
    print(f"  是否生产船 (y3): {int(y3.x)}")
    print(f"  是否生产火车 (y4): {int(y4.x)}")
    
    # 验证资源使用
    wood_used = 12*x1.x + 20*x2.x + 15*x3.x + 10*x4.x
    steel_used = 6*x1.x + 3*x2.x + 5*x3.x + 4*x4.x
    print(f"\n资源使用情况:")
    print(f"  木材使用: {wood_used} / 890 单位")
    print(f"  钢铁使用: {steel_used} / 500 单位")
else:
    print("未找到最优解。求解状态:", model.status)