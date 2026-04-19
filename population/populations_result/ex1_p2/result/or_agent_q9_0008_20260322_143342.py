# 完整Python代码
import gurobipy as gp
from gurobipy import GRB

# 参数与数据
# 资源约束右边常数
wood_limit = 890
steel_limit = 500

# 利润系数
profit = [5, 10, 8, 7]  # 对应卡车、飞机、船、火车

# 生产数量上限（用于二元变量关联约束）
max_production = [74, 44, 59, 89]  # 对应卡车、飞机、船、火车

# 模型构建
# 1. 创建模型
m = gp.Model("Toy_Production")

# 2. 定义决策变量
# x: 生产数量（非负整数）
x = m.addVars(4, lb=0, vtype=GRB.INTEGER, name="x")
# y: 是否生产（二元变量）
y = m.addVars(4, vtype=GRB.BINARY, name="y")

# 3. 设置目标函数
m.setObjective(gp.quicksum(profit[i] * x[i] for i in range(4)), GRB.MAXIMIZE)

# 4. 添加约束
# 木材资源约束: 12x1 + 20x2 + 15x3 + 10x4 <= 890
m.addConstr(12 * x[0] + 20 * x[1] + 15 * x[2] + 10 * x[3] <= wood_limit, "wood_constraint")

# 钢材资源约束: 6x1 + 3x2 + 5x3 + 4x4 <= 500
m.addConstr(6 * x[0] + 3 * x[1] + 5 * x[2] + 4 * x[3] <= steel_limit, "steel_constraint")

# 卡车与火车互斥逻辑: y1 + y4 <= 1
m.addConstr(y[0] + y[3] <= 1, "truck_train_exclusive")

# 船与飞机依赖逻辑: y3 <= y2
m.addConstr(y[2] <= y[1], "ship_depend_on_plane")

# 船的数量不超过火车的数量: x3 <= x4
m.addConstr(x[2] <= x[3], "ship_le_train")

# 生产数量与二元变量关联约束: 
# x1 <= 74*y1, x1 >= y1
# x2 <= 44*y2, x2 >= y2
# x3 <= 59*y3, x3 >= y3
# x4 <= 89*y4, x4 >= y4
for i in range(4):
    m.addConstr(x[i] <= max_production[i] * y[i], f"upper_bound_{i}")
    m.addConstr(x[i] >= y[i], f"lower_bound_{i}")

# 求解与输出
# 1. 求解模型
m.optimize()

# 2. 输出结果
if m.status == GRB.OPTIMAL:
    print("最优目标值 Z =", m.objVal)
    print("生产数量:")
    product_names = ["卡车", "飞机", "船", "火车"]
    for i in range(4):
        print(f"  {product_names[i]} (x{i+1}) = {x[i].x}")
    print("是否生产:")
    for i in range(4):
        print(f"  {product_names[i]} (y{i+1}) = {int(y[i].x)}")
else:
    print("未找到最优解")
    print("求解状态:", m.status)