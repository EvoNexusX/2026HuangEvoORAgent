from gurobipy import *

try:
    # 定义工人和任务集合
    workers = ['I', 'II', 'III', 'IV', 'V']
    tasks = ['A', 'B', 'C', 'D']
    
    # 定义每个工人完成每项任务的时间（小时）
    cost = {
        ('I', 'A'): 9, ('I', 'B'): 4, ('I', 'C'): 3, ('I', 'D'): 7,
        ('II', 'A'): 4, ('II', 'B'): 6, ('II', 'C'): 5, ('II', 'D'): 6,
        ('III', 'A'): 5, ('III', 'B'): 4, ('III', 'C'): 7, ('III', 'D'): 5,
        ('IV', 'A'): 7, ('IV', 'B'): 5, ('IV', 'C'): 2, ('IV', 'D'): 3,
        ('V', 'A'): 10, ('V', 'B'): 6, ('V', 'C'): 7, ('V', 'D'): 4
    }
    
    # 创建模型
    model = Model("Assignment_Problem")
    
    # 创建决策变量
    x = {}
    for i in workers:
        for j in tasks:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # 设置目标函数：最小化总工作时间
    model.setObjective(quicksum(cost[i, j] * x[i, j] for i in workers for j in tasks), GRB.MINIMIZE)
    
    # 添加约束：每项任务恰好由一名工人完成
    for j in tasks:
        model.addConstr(quicksum(x[i, j] for i in workers) == 1, name=f"task_{j}")
    
    # 添加约束：每名工人至多被分配一项任务
    for i in workers:
        model.addConstr(quicksum(x[i, j] for j in tasks) <= 1, name=f"worker_{i}")
    
    # 求解模型
    model.optimize()
    
    # 输出结果
    if model.status == GRB.OPTIMAL:
        print(f"最优总工作时间: {model.objVal} 小时")
        print("分配方案:")
        for i in workers:
            for j in tasks:
                if x[i, j].X > 0.5:
                    print(f"  工人 {i} -> 任务 {j}，时间: {cost[i, j]} 小时")
        # 输出未被选中的工人
        unused = [i for i in workers if all(x[i, j].X < 0.5 for j in tasks)]
        if unused:
            print(f"未被选中的工人: {unused}")
    else:
        print("未找到最优解")

except GurobiError as e:
    print(f"Gurobi错误: {e}")
except Exception as e:
    print(f"其他错误: {e}")
finally:
    model.dispose()