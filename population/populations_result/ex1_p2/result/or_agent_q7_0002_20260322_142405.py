import gurobipy as gp
from gurobipy import GRB

# 仓库和港口的索引集合
I = range(6)  # 0:Verona, 1:Perugia, 2:Rome, 3:Pescara, 4:Taranto, 5:Lamezia
J = range(5)  # 0:Genoa, 1:Venice, 2:Ancona, 3:Naples, 4:Bari

# 仓库库存量
s = [10, 12, 20, 24, 18, 40]

# 港口需求量
d = [20, 15, 25, 33, 21]

# 距离矩阵（公里）：dist[i][j] 表示从仓库i到港口j的距离
dist = [
    [290, 115, 355, 715, 810],   # Verona
    [380, 340, 165, 380, 610],   # Perugia
    [505, 530, 285, 220, 450],   # Rome
    [655, 450, 155, 240, 315],   # Pescara
    [1010, 840, 550, 305, 95],   # Taranto
    [1072, 1097, 747, 372, 333]  # Lamezia
]

# 创建模型
m = gp.Model("Transportation")

# 定义决策变量
x = m.addVars(I, J, vtype=GRB.INTEGER, name="x")
y = m.addVars(I, J, vtype=GRB.INTEGER, name="y")

# 设置目标函数：最小化总运输成本
m.setObjective(gp.quicksum(30 * dist[i][j] * y[i, j] for i in I for j in J), GRB.MINIMIZE)

# 添加约束
# 供应约束：每个仓库运出的集装箱总数不超过其库存量
supply_constrs = m.addConstrs((gp.quicksum(x[i, j] for j in J) <= s[i] for i in I), name="supply")

# 需求约束：每个港口接收的集装箱总数等于其需求量
demand_constrs = m.addConstrs((gp.quicksum(x[i, j] for i in I) == d[j] for j in J), name="demand")

# 卡车容量约束：每条路线上的集装箱数量不超过卡车运力的两倍
truck_constrs = m.addConstrs((x[i, j] <= 2 * y[i, j] for i in I for j in J), name="truck")

# 求解模型
m.optimize()

# 输出结果
if m.status == GRB.OPTIMAL:
    print(f"最优总成本: {m.objVal:.2f}")
    print("\n运输方案 (x_ij):")
    for i in I:
        for j in J:
            if x[i, j].x > 0:
                print(f"  从仓库{i+1}到港口{j+1}: {int(x[i, j].x)} 个集装箱")
    print("\n卡车使用数量 (y_ij):")
    for i in I:
        for j in J:
            if y[i, j].x > 0:
                print(f"  从仓库{i+1}到港口{j+1}: {int(y[i, j].x)} 辆卡车")
else:
    print("未找到最优解")