import gurobipy as gp
from gurobipy import GRB

def solve_transportation_problem():
    """
    解决意大利运输公司的空集装箱运输优化问题
    返回最优总成本
    """
    try:
        # 1. 创建模型
        model = gp.Model("EmptyContainerTransport")
        
        # 2. 定义数据
        # 仓库集合和供应量
        warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
        I = range(len(warehouses))
        supply = {
            0: 10,  # Verona
            1: 12,  # Perugia
            2: 20,  # Rome
            3: 24,  # Pescara
            4: 18,  # Taranto
            5: 40   # Lamezia
        }
        
        # 港口集合和需求量
        ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]
        J = range(len(ports))
        demand = {
            0: 20,  # Genoa
            1: 15,  # Venice
            2: 25,  # Ancona
            3: 33,  # Naples
            4: 21   # Bari
        }
        
        # 距离矩阵 (公里)
        distance = [
            # Genoa, Venice, Ancona, Naples, Bari
            [290, 115, 355, 715, 810],   # Verona
            [380, 340, 165, 380, 610],   # Perugia
            [505, 530, 285, 220, 450],   # Rome
            [655, 450, 155, 240, 315],   # Pescara
            [1010, 840, 550, 305, 95],   # Taranto
            [1072, 1097, 747, 372, 333]  # Lamezia
        ]
        
        # 单位成本 (欧元/公里·卡车)
        c = 30
        
        # 3. 创建决策变量
        # x[i,j]: 从仓库i运往港口j的集装箱数量
        # y[i,j]: 从仓库i到港口j的卡车数量
        x = model.addVars(I, J, vtype=GRB.INTEGER, name="x", lb=0)
        y = model.addVars(I, J, vtype=GRB.INTEGER, name="y", lb=0)
        
        # 4. 设置目标函数：最小化总运输成本
        objective = gp.quicksum(c * distance[i][j] * y[i, j] for i in I for j in J)
        model.setObjective(objective, GRB.MINIMIZE)
        
        # 5. 添加约束
        
        # 供应约束：每个仓库运出的集装箱不超过供应量
        for i in I:
            model.addConstr(gp.quicksum(x[i, j] for j in J) <= supply[i], 
                          name=f"supply_{warehouses[i]}")
        
        # 需求约束：每个港口收到的集装箱等于需求量
        for j in J:
            model.addConstr(gp.quicksum(x[i, j] for i in I) == demand[j], 
                          name=f"demand_{ports[j]}")
        
        # 卡车容量约束：每辆卡车最多装载2个集装箱
        for i in I:
            for j in J:
                model.addConstr(x[i, j] <= 2 * y[i, j], 
                              name=f"truck_capacity_{i}_{j}")
        
        # 6. 设置求解参数
        model.setParam('OutputFlag', 1)  # 显示求解过程
        model.setParam('MIPGap', 0.01)   # 设置最优间隙为1%
        
        # 7. 求解模型
        model.optimize()
        
        # 8. 输出结果
        if model.status == GRB.OPTIMAL:
            print(f"优化成功！最优总成本: {model.objVal:.2f} 欧元")
            print("\n运输方案详情 (仅显示非零运输路径):")
            print("-" * 70)
            print(f"{'仓库':<10} {'港口':<10} {'集装箱数':<12} {'卡车数':<12} {'距离(km)':<12} {'成本(欧元)':<12}")
            print("-" * 70)
            
            total_containers = 0
            for i in I:
                for j in J:
                    if x[i, j].x > 0:
                        containers = x[i, j].x
                        trucks = y[i, j].x
                        dist = distance[i][j]
                        cost = c * dist * trucks
                        total_containers += containers
                        print(f"{warehouses[i]:<10} {ports[j]:<10} {containers:<12.0f} {trucks:<12.0f} {dist:<12} {cost:<12.2f}")
            
            print("-" * 70)
            print(f"总计运输集装箱: {total_containers}")
            
            # 验证供需
            print("\n供需验证:")
            for i in I:
                shipped = sum(x[i, j].x for j in J)
                print(f"{warehouses[i]}: 供应={supply[i]}, 运出={shipped}, 剩余={supply[i]-shipped}")
            
            for j in J:
                received = sum(x[i, j].x for i in I)
                print(f"{ports[j]}: 需求={demand[j]}, 收到={received}, 缺口={demand[j]-received}")
            
            return model.objVal
            
        else:
            print(f"未找到最优解。状态代码: {model.status}")
            return None
            
    except gp.GurobiError as e:
        print(f"Gurobi错误: {e}")
        return None
    except Exception as e:
        print(f"其他错误: {e}")
        return None

# 运行求解函数
if __name__ == "__main__":
    optimal_cost = solve_transportation_problem()
    if optimal_cost is not None:
        print(f"\n最优目标值: {optimal_cost:.2f} 欧元")