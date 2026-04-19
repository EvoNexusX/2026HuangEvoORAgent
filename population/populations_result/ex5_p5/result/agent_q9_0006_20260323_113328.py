import gurobipy as gp

# 创建模型
model = gp.Model("Haus_Toys_Production")

# 参数设置
profit = {"Truck": 5, "Airplane": 10, "Boat": 8, "Train": 7}
wood_use = {"Truck": 12, "Airplane": 20, "Boat": 15, "Train": 10}
steel_use = {"Truck": 6, "Airplane": 3, "Boat": 5, "Train": 4}
wood_total = 890
steel_total = 500

# 计算上界
U_T = min(wood_total // wood_use["Truck"], steel_total // steel_use["Truck"])
U_A = min(wood_total // wood_use["Airplane"], steel_total // steel_use["Airplane"])
U_B = min(wood_total // wood_use["Boat"], steel_total // steel_use["Boat"])
U_R = min(wood_total // wood_use["Train"], steel_total // steel_use["Train"])

# 决策变量
x_T = model.addVar(lb=0, ub=U_T, vtype=gp.GRB.INTEGER, name="x_T")
x_A = model.addVar(lb=0, ub=U_A, vtype=gp.GRB.INTEGER, name="x_A")
x_B = model.addVar(lb=0, ub=U_B, vtype=gp.GRB.INTEGER, name="x_B")
x_R = model.addVar(lb=0, ub=U_R, vtype=gp.GRB.INTEGER, name="x_R")
y_T = model.addVar(vtype=gp.GRB.BINARY, name="y_T")
y_B = model.addVar(vtype=gp.GRB.BINARY, name="y_B")

# 设置目标函数
model.setObjective(profit["Truck"] * x_T + profit["Airplane"] * x_A + 
                   profit["Boat"] * x_B + profit["Train"] * x_R, 
                   gp.GRB.MAXIMIZE)

# 添加约束条件
model.addConstr(wood_use["Truck"] * x_T + wood_use["Airplane"] * x_A + 
                wood_use["Boat"] * x_B + wood_use["Train"] * x_R <= wood_total, 
                "Wood_Constraint")
model.addConstr(steel_use["Truck"] * x_T + steel_use["Airplane"] * x_A + 
                steel_use["Boat"] * x_B + steel_use["Train"] * x_R <= steel_total, 
                "Steel_Constraint")
model.addConstr(x_T <= U_T * y_T, "Truck_Indicator_Upper")
model.addConstr(x_T >= y_T, "Truck_Indicator_Lower")
model.addConstr(x_B <= U_B * y_B, "Boat_Indicator_Upper")
model.addConstr(x_B >= y_B, "Boat_Indicator_Lower")
model.addConstr(x_R <= U_R * (1 - y_T), "Truck_Train_Exclusive")
model.addConstr(x_A >= y_B, "Boat_Airplane_Link")
model.addConstr(x_B <= x_R, "Boat_Train_Quantity")

# 求解模型
model.optimize()

# 输出结果
if model.status == gp.GRB.OPTIMAL:
    print(f'最优利润: ${model.objVal:.2f}')
    print(f'卡车生产数量: {x_T.x}')
    print(f'飞机生产数量: {x_A.x}')
    print(f'船生产数量: {x_B.x}')
    print(f'火车生产数量: {x_R.x}')
    print(f'是否生产卡车 (y_T): {y_T.x}')
    print(f'是否生产船 (y_B): {y_B.x}')
else:
    print('未找到最优解')