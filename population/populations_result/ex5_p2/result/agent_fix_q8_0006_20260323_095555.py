# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

def solve_assignment():
    # 定义数据
    I = ["I", "II", "III", "IV", "V"]  # 工人集合
    J = ["A", "B", "C", "D"]           # 任务集合
    
    # 成本数据（工时）
    c = {
        ("I", "A"): 9, ("I", "B"): 4, ("I", "C"): 3, ("I", "D"): 7,
        ("II", "A"): 4, ("II", "B"): 6, ("II", "C"): 5, ("II", "D"): 6,
        ("III", "A"): 5, ("III", "B"): 4, ("III", "C"): 7, ("III", "D"): 5,
        ("IV", "A"): 7, ("IV", "B"): 5, ("IV", "C"): 2, ("IV", "D"): 3,
        ("V", "A"): 10, ("V", "B"): 6, ("V", "C"): 7, ("V", "D"): 4
    }
    
    # 创建模型
    m = gp.Model("assignment")
    
    # 添加变量
    x = m.addVars(I, J, vtype=GRB.BINARY, name="x")
    
    # 设置目标函数
    m.setObjective(gp.quicksum(c[i, j] * x[i, j] for i in I for j in J), GRB.MINIMIZE)
    
    # 添加约束：每个任务恰好由一个工人完成
    m.addConstrs((x.sum('*', j) == 1 for j in J), name="task")
    
    # 添加约束：每个工人至多分配一个任务
    m.addConstrs((x.sum(i, '*') <= 1 for i in I), name="worker")
    
    # 求解模型
    m.optimize()
    
    # 输出结果
    if m.status == GRB.OPTIMAL:
        print("最优总时间:", m.objVal)
        print("\n分配方案:")
        for i in I:
            for j in J:
                if x[i, j].X > 0.5:
                    print(f"工人 {i} -> 任务 {j} (工时: {c[i, j]})")
    else:
        print("未找到最优解")
        print("求解状态:", m.status)

if __name__ == "__main__":
    solve_assignment()