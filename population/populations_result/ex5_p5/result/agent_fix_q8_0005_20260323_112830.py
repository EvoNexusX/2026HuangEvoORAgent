# -*- coding: utf-8 -*-
from gurobipy import Model, GRB, quicksum

workers = [0, 1, 2, 3, 4]
tasks = [0, 1, 2, 3]
cost = [
    [9, 4, 3, 7],
    [4, 6, 5, 6],
    [5, 4, 7, 5],
    [7, 5, 2, 3],
    [10, 6, 7, 4]
]

model = Model("Assignment_Problem")

x = model.addVars(workers, tasks, vtype=GRB.BINARY, name="x")

model.setObjective(quicksum(cost[i][j] * x[i, j] for i in workers for j in tasks), GRB.MINIMIZE)

for j in tasks:
    model.addConstr(quicksum(x[i, j] for i in workers) == 1, name=f"task_{j}")

for i in workers:
    model.addConstr(quicksum(x[i, j] for j in tasks) <= 1, name=f"worker_{i}")

model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"Optimal total time: {model.objVal}")
    print("Assignment plan:")
    for i in workers:
        for j in tasks:
            if x[i, j].X > 0.5:
                worker_names = ['I', 'II', 'III', 'IV', 'V']
                task_names = ['A', 'B', 'C', 'D']
                print(f"Worker {worker_names[i]} -> Task {task_names[j]} (time: {cost[i][j]})")
else:
    print("No optimal solution found")