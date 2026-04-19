from gurobipy import Model, GRB, quicksum

# 步骤1：初始化模型
model = Model('assignment')

# 步骤2：定义集合和参数
workers = [1, 2, 3, 4, 5]
tasks = ['A', 'B', 'C', 'D']

time = {
    (1, 'A'): 9, (1, 'B'): 4, (1, 'C'): 3, (1, 'D'): 7,
    (2, 'A'): 4, (2, 'B'): 6, (2, 'C'): 5, (2, 'D'): 6,
    (3, 'A'): 5, (3, 'B'): 4, (3, 'C'): 7, (3, 'D'): 5,
    (4, 'A'): 7, (4, 'B'): 5, (4, 'C'): 2, (4, 'D'): 3,
    (5, 'A'): 10, (5, 'B'): 6, (5, 'C'): 7, (5, 'D'): 4
}

# 步骤3：定义决策变量
x = model.addVars(workers, tasks, vtype=GRB.BINARY, name='x')

# 步骤4：设置目标函数
model.setObjective(
    quicksum(time[i, j] * x[i, j] for i in workers for j in tasks),
    GRB.MINIMIZE
)

# 步骤5：添加约束条件
# 每项任务必须由恰好一名工人完成
for j in tasks:
    model.addConstr(
        quicksum(x[i, j] for i in workers) == 1,
        name=f'task_{j}'
    )

# 每名工人最多承担一项任务
for i in workers:
    model.addConstr(
        quicksum(x[i, j] for j in tasks) <= 1,
        name=f'worker_{i}'
    )

# 步骤6：求解模型
model.optimize()

# 步骤7：结果输出
if model.status == GRB.OPTIMAL:
    print(f'最优总工作时间: {model.objVal} 小时')
    print('\n分配方案:')
    for i in workers:
        assigned = False
        for j in tasks:
            if x[i, j].x > 0.5:
                print(f'工人 {i} -> 任务 {j} (时间: {time[i, j]} 小时)')
                assigned = True
        if not assigned:
            print(f'工人 {i} -> 未分配任务')
    print('\n任务分配详情:')
    for j in tasks:
        for i in workers:
            if x[i, j].x > 0.5:
                print(f'任务 {j} -> 工人 {i}')
else:
    print('未找到最优解')