import gurobipy as gp
from gurobipy import GRB

# 参数与数据
wood_limit = 890
steel_limit = 500
profit = [5, 10, 8, 7]  # 卡车、飞机、船、火车
max_production = [74, 44, 59, 89]  # 卡车、飞机、船、火车

# 创建模型
m = gp.Model("Toy_Production_Optimization")

# 定义决策变量
x = m.addVars(4, lb=0, vtype=GRB.INTEGER, name="x")
y = m.addVars(4, vtype=GRB.BINARY, name="y")

# 设置目标函数
m.setObjective(
    gp.quicksum(profit[i] * x[i] for i in range(4)), 
    GRB.MAXIMIZE
)

# 添加约束条件
# 1. 木材资源约束
m.addConstr(
    12 * x[0] + 20 * x[1] + 15 * x[2] + 10 * x[3] <= wood_limit,
    "wood_constraint"
)

# 2. 钢材资源约束
m.addConstr(
    6 * x[0] + 3 * x[1] + 5 * x[2] + 4 * x[3] <= steel_limit,
    "steel_constraint"
)

# 3. 卡车与火车互斥逻辑
m.addConstr(y[0] + y[3] <= 1, "truck_train_exclusive")

# 4. 船与飞机依赖逻辑
m.addConstr(y[2] <= y[1], "ship_depend_on_plane")

# 5. 船的数量不超过火车的数量
m.addConstr(x[2] <= x[3], "ship_le_train")

# 6-9. 生产数量与二元变量关联约束
for i in range(4):
    m.addConstr(x[i] <= max_production[i] * y[i], f"upper_bound_{i}")
    m.addConstr(x[i] >= y[i], f"lower_bound_{i}")

# 求解模型
m.optimize()

# 输出结果
if m.status == GRB.OPTIMAL:
    print(f"最优目标值 Z = {m.objVal:.2f}")
    print("\n生产数量:")
    product_names = ["卡车", "飞机", "船", "火车"]
    for i in range(4):
        print(f"  {product_names[i]}: {int(x[i].x)} 单位")
    
    print("\n生产决策:")
    for i in range(4):
        decision = "生产" if y[i].x > 0.5 else "不生产"
        print(f"  {product_names[i]}: {decision}")
    
    # 详细输出资源使用情况
    wood_used = 12 * x[0].x + 20 * x[1].x + 15 * x[2].x + 10 * x[3].x
    steel_used = 6 * x[0].x + 3 * x[1].x + 5 * x[2].x + 4 * x[3].x
    
    print(f"\n资源使用情况:")
    print(f"  木材: {wood_used:.1f} / {wood_limit} ({wood_used/wood_limit*100:.1f}%)")
    print(f"  钢材: {steel_used:.1f} / {steel_limit} ({steel_used/steel_limit*100:.1f}%)")
    
    # 检查特殊约束
    print(f"\n特殊约束检查:")
    print(f"  卡车和火车互斥: {'满足' if y[0].x + y[3].x <= 1 else '违反'}")
    print(f"  船依赖飞机: {'满足' if y[2].x <= y[1].x else '违反'}")
    print(f"  船不超过火车: {'满足' if x[2].x <= x[3].x else '违反'}")
    
else:
    print("未找到最优解")
    print(f"求解状态: {m.status}")