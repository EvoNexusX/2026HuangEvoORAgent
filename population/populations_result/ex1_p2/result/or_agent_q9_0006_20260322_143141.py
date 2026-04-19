# 完整Python代码
import gurobipy as gp
from gurobipy import GRB

# 参数与数据
wood_limit = 890
steel_limit = 500
profit = [5, 10, 8, 7]  # 对应卡车、飞机、船、火车
max_production = [74, 44, 59, 89]  # 对应卡车、飞机、船、火车

# 模型构建
m = gp.Model("ToyProduction")

# 定义决策变量
x = m.addVars(4, lb=0, vtype=GRB.INTEGER, name="x")  # 生产数量
y = m.addVars(4, vtype=GRB.BINARY, name="y")          # 是否生产

# 设置目标函数
m.setObjective(gp.quicksum(profit[i] * x[i] for i in range(4)), GRB.MAXIMIZE)

# 添加约束
# 木材资源约束: 12x1 + 20x2 + 15x3 + 10x4 ≤ 890
m.addConstr(12 * x[0] + 20 * x[1] + 15 * x[2] + 10 * x[3] <= wood_limit, "Wood")

# 钢材资源约束: 6x1 + 3x2 + 5x3 + 4x4 ≤ 500
m.addConstr(6 * x[0] + 3 * x[1] + 5 * x[2] + 4 * x[3] <= steel_limit, "Steel")

# 卡车与火车互斥: y1 + y4 ≤ 1
m.addConstr(y[0] + y[3] <= 1, "Truck_Train_Exclusive")

# 船依赖飞机: y3 ≤ y2
m.addConstr(y[2] <= y[1], "Ship_Depends_on_Plane")

# 船的数量不超过火车的数量: x3 ≤ x4
m.addConstr(x[2] <= x[3], "Ship_leq_Train")

# 生产数量与二元变量关联
for i in range(4):
    m.addConstr(x[i] <= max_production[i] * y[i], f"Upper_Bound_{i}")
    m.addConstr(x[i] >= y[i], f"Lower_Bound_{i}")

# 求解模型
m.optimize()

# 输出结果
if m.status == GRB.OPTIMAL:
    print(f"最优目标值 Z = {m.objVal:.2f}")
    print("生产数量:")
    toys = ["卡车", "飞机", "船", "火车"]
    for i in range(4):
        print(f"  {toys[i]}: {x[i].x:.0f} 单位")
    print("是否生产:")
    for i in range(4):
        print(f"  {toys[i]}: {int(y[i].x)}")
else:
    print("未找到最优解")