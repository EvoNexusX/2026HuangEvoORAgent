import gurobipy as gp
from gurobipy import GRB

def solve_container_transport():
    """
    Solves the container transportation optimization problem using Gurobi.
    """
    try:
        # --- 1. 初始化与模型创建 ---
        model = gp.Model("ContainerTransport")
        
        # --- 2. 数据准备 ---
        # 仓库集合 (索引对应: 0-5)
        warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
        I = list(range(len(warehouses)))
        
        # 港口集合 (索引对应: 0-4)
        ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]
        J = list(range(len(ports)))
        
        # 供应量 (空集装箱数量)
        supply = [10, 12, 20, 24, 18, 40]
        
        # 需求量 (集装箱需求)
        demand = [20, 15, 25, 33, 21]
        
        # 距离矩阵 (公里) - 行对应仓库，列对应港口
        distances = [
            [290, 115, 355, 715, 810],   # Verona
            [380, 340, 165, 380, 610],   # Perugia
            [505, 530, 285, 220, 450],   # Rome
            [655, 450, 155, 240, 315],   # Pescara
            [1010, 840, 550, 305, 95],   # Taranto
            [1072, 1097, 747, 372, 333]  # Lamezia
        ]
        
        # 运输成本参数 (欧元/公里)
        cost_per_km = 30
        
        # --- 3. 决策变量定义 ---
        # x_ij: 从仓库i到港口j的卡车数量 (整数)
        x = model.addVars(I, J, vtype=GRB.INTEGER, name="trucks")
        # y_ij: 从仓库i到港口j的集装箱数量 (整数)
        y = model.addVars(I, J, vtype=GRB.INTEGER, name="containers")
        
        # --- 4. 目标函数设置 ---
        # 最小化总运输成本 = sum(cost_per_km * distance_ij * trucks_ij)
        model.setObjective(
            gp.quicksum(cost_per_km * distances[i][j] * x[i, j] 
                       for i in I for j in J),
            GRB.MINIMIZE
        )
        
        # --- 5. 约束条件添加 ---
        # 供应约束: 每个仓库运出的集装箱总数不超过其供应量
        model.addConstrs(
            (y.sum(i, '*') <= supply[i] for i in I),
            name="supply"
        )
        
        # 需求约束: 每个港口收到的集装箱总数必须满足其需求
        model.addConstrs(
            (y.sum('*', j) == demand[j] for j in J),
            name="demand"
        )
        
        # 卡车容量约束: 每辆卡车最多装载2个集装箱
        model.addConstrs(
            (y[i, j] <= 2 * x[i, j] for i in I for j in J),
            name="capacity"
        )
        
        # --- 6. 模型求解 ---
        model.optimize()
        
        # --- 7. 结果输出 ---
        print("=" * 60)
        print("容器运输优化问题 - 求解结果")
        print("=" * 60)
        
        if model.status == GRB.OPTIMAL:
            print(f"\n最优总成本: {model.objVal:,.2f} 欧元")
            
            # 按仓库汇总输出
            print("\n" + "=" * 60)
            print("按仓库汇总运输情况:")
            print("=" * 60)
            for i in I:
                total_containers = sum(y[i, j].x for j in J)
                total_trucks = sum(x[i, j].x for j in J)
                if total_containers > 0:
                    print(f"\n{warehouses[i]} (供应量: {supply[i]}):")
                    print(f"  运出集装箱总数: {total_containers}")
                    print(f"  使用卡车总数: {total_trucks}")
                    for j in J:
                        if y[i, j].x > 0:
                            cost = cost_per_km * distances[i][j] * x[i, j].x
                            print(f"  → {ports[j]}: {int(y[i, j].x)} 个集装箱, "
                                  f"{int(x[i, j].x)} 辆卡车, 成本: {cost:,.2f} 欧元")
            
            # 按港口汇总输出
            print("\n" + "=" * 60)
            print("按港口汇总收货情况:")
            print("=" * 60)
            for j in J:
                total_received = sum(y[i, j].x for i in I)
                print(f"\n{ports[j]} (需求量: {demand[j]}):")
                print(f"  收到集装箱总数: {total_received}")
                for i in I:
                    if y[i, j].x > 0:
                        print(f"  ← {warehouses[i]}: {int(y[i, j].x)} 个集装箱")
            
            # 详细的运输方案表格
            print("\n" + "=" * 60)
            print("详细运输方案:")
            print("=" * 60)
            print("\n{:<10} {:<10} {:<10} {:<10} {:<15}".format(
                "仓库", "港口", "集装箱", "卡车", "成本(欧元)"
            ))
            print("-" * 60)
            total_containers_shipped = 0
            total_trucks_used = 0
            
            for i in I:
                for j in J:
                    if y[i, j].x > 0:
                        containers = int(y[i, j].x)
                        trucks = int(x[i, j].x)
                        cost = cost_per_km * distances[i][j] * trucks
                        total_containers_shipped += containers
                        total_trucks_used += trucks
                        
                        print("{:<10} {:<10} {:<10} {:<10} {:<15,.2f}".format(
                            warehouses[i], ports[j], containers, trucks, cost
                        ))
            
            print("-" * 60)
            print(f"总计: {total_containers_shipped} 个集装箱, "
                  f"{total_trucks_used} 辆卡车")
            
        elif model.status == GRB.INFEASIBLE:
            print("问题不可行！请检查约束条件。")
        elif model.status == GRB.UNBOUNDED:
            print("问题无界！")
        else:
            print(f"求解状态: {model.status}")
            
    except gp.GurobiError as e:
        print(f"Gurobi错误: {e}")
    except Exception as e:
        print(f"程序错误: {e}")

if __name__ == "__main__":
    solve_container_transport()