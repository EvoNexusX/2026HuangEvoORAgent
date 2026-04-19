import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("EmptyContainerTransport")

# 定义数据
# 仓库：Verona, Perugia, Rome, Pescara, Taranto, Lamezia
warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
I = range(len(warehouses))
supply = [10, 12, 20, 24, 18, 40]

# 港口：Genoa, Venice, Ancona, Naples, Bari
ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]
J = range(len(ports))
demand = [20, 15, 25, 33, 21]

# 距离矩阵（公里）
dist = [
    [290, 115, 355, 715, 810],   # Verona
    [380, 340, 165, 380, 610],   # Perugia
    [505, 530, 285, 220, 450],   # Rome
    [655, 450, 155, 240, 315],   # Pescara
    [1010, 840, 550, 305, 95],   # Taranto
    [1072, 1097, 747, 372, 333]  # Lamezia
]

# 成本系数：30欧元/公里
c = 30

# 创建变量
x = model.addVars(I, J, vtype=GRB.INTEGER, name="x")
y = model.addVars(I, J, vtype=GRB.INTEGER, name="y")

# 设置目标函数：最小化总运输成本
model.setObjective(gp.quicksum(c * dist[i][j] * y[i,j] for i in I for j in J), GRB.MINIMIZE)

# 添加约束
# 供应约束
supply_constrs = model.addConstrs((x.sum(i, '*') <= supply[i] for i in I), name="supply")

# 需求约束
demand_constrs = model.addConstrs((x.sum('*', j) == demand[j] for j in J), name="demand")

# 卡车容量约束
truck_constrs = model.addConstrs((x[i,j] <= 2 * y[i,j] for i in I for j in J), name="truck_capacity")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"最优总成本: {model.objVal:.2f} 欧元\n")
    
    print("运输方案（仅显示非零路径）:")
    print("-" * 60)
    total_cost = 0
    for i in I:
        for j in J:
            if x[i,j].X > 0:
                path_cost = c * dist[i][j] * y[i,j].X
                total_cost += path_cost
                print(f"{warehouses[i]} -> {ports[j]}: {int(x[i,j].X)} 个集装箱, {int(y[i,j].X)} 辆卡车, 成本: {path_cost:.2f} 欧元")
    
    print("-" * 60)
    print(f"验证总成本: {total_cost:.2f} 欧元")
    
    # 验证约束
    print("\n约束验证:")
    for i in I:
        shipped = sum(x[i,j].X for j in J)
        print(f"{warehouses[i]}: 供应 {supply[i]}, 实际运出 {shipped}")
    
    for j in J:
        received = sum(x[i,j].X for i in I)
        print(f"{ports[j]}: 需求 {demand[j]}, 实际接收 {received}")
        
elif model.status == GRB.INFEASIBLE:
    print("模型不可行")
    model.computeIIS()
    model.write("model.ilp")
    print("不可行约束已写入 model.ilp 文件")
elif model.status == GRB.UNBOUNDED:
    print("模型无界")
elif model.status == GRB.TIME_LIMIT:
    print("达到时间限制")
    print(f"当前可行解的目标值: {model.objVal:.2f}")
else:
    print(f"求解终止，状态码: {model.status}")