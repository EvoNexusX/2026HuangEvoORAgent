import gurobipy as gp
from gurobipy import GRB

def main():
    # 数据定义
    warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
    ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]
    
    I = range(len(warehouses))  # 仓库索引
    J = range(len(ports))       # 港口索引
    
    s = [10, 12, 20, 24, 18, 40]  # 供应量
    d = [20, 15, 25, 33, 21]      # 需求量
    
    # 距离矩阵（公里），来自问题描述
    dist = [
        [290, 115, 355, 715, 810],
        [380, 340, 165, 380, 610],
        [505, 530, 285, 220, 450],
        [655, 450, 155, 240, 315],
        [1010, 840, 550, 305, 95],
        [1072, 1097, 747, 372, 333]
    ]
    
    # 创建模型
    model = gp.Model("ContainerTransport")
    
    # 定义变量
    x = model.addVars(I, J, lb=0.0, name="x")
    
    # 设置目标函数：最小化总运输成本
    model.setObjective(
        gp.quicksum(30 * dist[i][j] * x[i, j] for i in I for j in J),
        GRB.MINIMIZE
    )
    
    # 添加供应约束
    for i in I:
        model.addConstr(
            gp.quicksum(x[i, j] for j in J) <= s[i],
            name=f"Supply_{warehouses[i]}"
        )
    
    # 添加需求约束
    for j in J:
        model.addConstr(
            gp.quicksum(x[i, j] for i in I) == d[j],
            name=f"Demand_{ports[j]}"
        )
    
    # 求解模型
    model.optimize()
    
    # 输出结果
    if model.status == GRB.OPTIMAL:
        print(f"最小总成本: {model.objVal:.2f} 欧元")
        print("\n运输方案（仅显示运输量>0的路径）:")
        for i in I:
            for j in J:
                if x[i, j].x > 1e-6:
                    print(f"  从 {warehouses[i]} 到 {ports[j]}: {x[i, j].x:.0f} 个集装箱")
        
        print("\n各仓库剩余集装箱:")
        for i in I:
            shipped = sum(x[i, j].x for j in J)
            remaining = s[i] - shipped
            print(f"  {warehouses[i]}: {remaining:.0f}")
    else:
        print("未找到最优解")

if __name__ == "__main__":
    main()