import gurobipy as gp
from gurobipy import GRB

def solve_production_model():
    # 创建模型
    model = gp.Model("Production_Planning")
    
    # 决策变量
    x1 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x1")  # 卡车数量
    x2 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x2")  # 飞机数量
    x3 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x3")  # 船数量
    x4 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x4")  # 火车数量
    
    y1 = model.addVar(vtype=GRB.BINARY, name="y1")  # 是否生产卡车
    y2 = model.addVar(vtype=GRB.BINARY, name="y2")  # 是否生产飞机
    y3 = model.addVar(vtype=GRB.BINARY, name="y3")  # 是否生产船
    y4 = model.addVar(vtype=GRB.BINARY, name="y4")  # 是否生产火车
    
    # 设置目标函数
    model.setObjective(5*x1 + 10*x2 + 8*x3 + 7*x4, GRB.MAXIMIZE)
    
    # 资源约束
    model.addConstr(12*x1 + 20*x2 + 15*x3 + 10*x4 <= 890, "resource_1")
    model.addConstr(6*x1 + 3*x2 + 5*x3 + 4*x4 <= 500, "resource_2")
    
    # 变量与二元变量关联约束
    model.addConstr(x1 >= y1, "link_x1_lower")
    model.addConstr(x1 <= 74*y1, "link_x1_upper")
    model.addConstr(x2 >= y2, "link_x2_lower")
    model.addConstr(x2 <= 44*y2, "link_x2_upper")
    model.addConstr(x3 >= y3, "link_x3_lower")
    model.addConstr(x3 <= 59*y3, "link_x3_upper")
    model.addConstr(x4 >= y4, "link_x4_lower")
    model.addConstr(x4 <= 89*y4, "link_x4_upper")
    
    # 逻辑约束
    model.addConstr(y1 + y4 <= 1, "exclusive_truck_train")  # 卡车与火车互斥
    model.addConstr(y3 <= y2, "ship_requires_airplane")      # 船生产则必须生产飞机
    model.addConstr(x3 <= x4, "ship_leq_train")              # 船的数量不超过火车数量
    
    # 求解设置
    model.Params.OutputFlag = 0  # 不输出求解过程信息
    
    # 求解
    model.optimize()
    
    # 结果输出
    if model.status == GRB.OPTIMAL:
        print(f"最大利润: ${model.objVal:.2f}")
        print("最优生产计划:")
        print(f"  卡车: {int(x1.x)} 个 (是否生产: {int(y1.x)})")
        print(f"  飞机: {int(x2.x)} 个 (是否生产: {int(y2.x)})")
        print(f"  船: {int(x3.x)} 个 (是否生产: {int(y3.x)})")
        print(f"  火车: {int(x4.x)} 个 (是否生产: {int(y4.x)})")
        print("\n资源使用情况:")
        wood_used = 12*x1.x + 20*x2.x + 15*x3.x + 10*x4.x
        steel_used = 6*x1.x + 3*x2.x + 5*x3.x + 4*x4.x
        print(f"  木材: {wood_used:.0f} / 890 单位")
        print(f"  钢材: {steel_used:.0f} / 500 单位")
    else:
        print("未找到最优解")
    
    return model

if __name__ == "__main__":
    solve_production_model()