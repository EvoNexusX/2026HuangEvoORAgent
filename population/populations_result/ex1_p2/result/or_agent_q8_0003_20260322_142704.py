import gurobipy as gp
from gurobipy import GRB

# 创建模型
m = gp.Model("Assignment_Problem")

# 参数与数据
num_workers = 5
num_tasks = 4

# 成本矩阵 c[i][j] 表示工人i完成任务j的小时数
c = [
    [9, 4, 3, 7],
    [4, 6, 5, 6],
    [5, 4, 7, 5],
    [7, 5, 2, 3],
    [10, 6, 7, 4]
]

# 定义决策变量：x[i][j]为二元变量
x = m.addVars(num_workers, num_tasks, vtype=GRB.BINARY, name="x")

# 设置目标函数：最小化总工作时间
m.setObjective(
    gp.quicksum(c[i][j] * x[i, j] for i in range(num_workers) for j in range(num_tasks)),
    GRB.MINIMIZE
)

# 添加约束
# 每个任务必须由恰好一个工人完成
for j in range(num_tasks):
    m.addConstr(
        gp.quicksum(x[i, j] for i in range(num_workers)) == 1,
        name=f"Task_{j}"
    )

# 每个工人最多完成一个任务
for i in range(num_workers):
    m.addConstr(
        gp.quicksum(x[i, j] for j in range(num_tasks)) <= 1,
        name=f"Worker_{i}"
    )

# 求解模型
m.optimize()

# 输出结果
if m.status == GRB.OPTIMAL:
    print(f"最优总工作时间: {m.ObjVal} 小时")
    print("任务分配情况:")
    for i in range(num_workers):
        for j in range(num_tasks):
            if x[i, j].X > 0.5:  # 由于是二元变量，值接近1即表示被选中
                # 注意：索引从0开始，转换为从1开始以便阅读
                worker_name = f"工人 {i+1}"
                task_name = chr(ord('A') + j)  # 任务A, B, C, D
                print(f"  {worker_name} 被分配给任务 {task_name}，工作时间: {c[i][j]} 小时")
else:
    print("未找到最优解")