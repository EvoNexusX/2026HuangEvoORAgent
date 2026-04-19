import gurobipy as gp
from gurobipy import GRB

def main():
    try:
        # 1. 模型初始化
        model = gp.Model("WorkerAssignment")
        
        # 2. 数据定义
        workers = [1, 2, 3, 4, 5]  # I, II, III, IV, V
        tasks = ['A', 'B', 'C', 'D']
        
        # 工时矩阵: workers[行] -> tasks[列]
        hours = {
            1: {'A': 9, 'B': 4, 'C': 3, 'D': 7},   # Worker I
            2: {'A': 4, 'B': 6, 'C': 5, 'D': 6},   # Worker II
            3: {'A': 5, 'B': 4, 'C': 7, 'D': 5},   # Worker III
            4: {'A': 7, 'B': 5, 'C': 2, 'D': 3},   # Worker IV
            5: {'A': 10, 'B': 6, 'C': 7, 'D': 4}   # Worker V
        }
        
        # 3. 决策变量: x[i,j] = 1 表示工人i被分配到任务j
        x = model.addVars(workers, tasks, vtype=GRB.BINARY, name="x")
        
        # 4. 目标函数: 最小化总工时
        model.setObjective(
            gp.quicksum(hours[i][j] * x[i,j] for i in workers for j in tasks),
            GRB.MINIMIZE
        )
        
        # 5. 约束条件
        # 每个任务必须由恰好一个工人完成
        for j in tasks:
            model.addConstr(gp.quicksum(x[i,j] for i in workers) == 1, f"task_{j}")
        
        # 每个工人最多完成一项任务
        for i in workers:
            model.addConstr(gp.quicksum(x[i,j] for j in tasks) <= 1, f"worker_{i}")
        
        # 6. 求解模型
        model.optimize()
        
        # 7. 结果处理
        if model.status == GRB.OPTIMAL:
            print(f"最优总工时: {model.objVal:.0f} 小时")
            print("\n最优分配方案:")
            assigned_workers = set()
            for j in tasks:
                for i in workers:
                    if x[i,j].x > 0.5:
                        # 转换工人编号为罗马数字
                        worker_names = ['I', 'II', 'III', 'IV', 'V']
                        print(f"  任务 {j} → 工人 {worker_names[i-1]} (工时: {hours[i][j]})")
                        assigned_workers.add(i)
            
            # 找出未分配的工人
            unassigned = [i for i in workers if i not in assigned_workers]
            if unassigned:
                worker_names = ['I', 'II', 'III', 'IV', 'V']
                print(f"\n未分配的工人: 工人 {worker_names[unassigned[0]-1]}")
                
        elif model.status == GRB.INFEASIBLE:
            print("模型无可行解")
        elif model.status == GRB.UNBOUNDED:
            print("模型无界")
        elif model.status == GRB.INF_OR_UNBD:
            print("模型不可行或无界")
        else:
            print(f"求解终止，状态码: {model.status}")
            
    except gp.GurobiError as e:
        print(f"Gurobi错误: {e}")
    except Exception as e:
        print(f"程序错误: {e}")
    finally:
        # 8. 释放资源
        if 'model' in locals():
            model.dispose()

if __name__ == "__main__":
    main()