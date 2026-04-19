# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

# Define worker and task sets
workers = ['I', 'II', 'III', 'IV', 'V']
tasks = ['A', 'B', 'C', 'D']

# Define working hours parameters
cost = {
    ('I', 'A'): 9, ('I', 'B'): 4, ('I', 'C'): 3, ('I', 'D'): 7,
    ('II', 'A'): 4, ('II', 'B'): 6, ('II', 'C'): 5, ('II', 'D'): 6,
    ('III', 'A'): 5, ('III', 'B'): 4, ('III', 'C'): 7, ('III', 'D'): 5,
    ('IV', 'A'): 7, ('IV', 'B'): 5, ('IV', 'C'): 2, ('IV', 'D'): 3,
    ('V', 'A'): 10, ('V', 'B'): 6, ('V', 'C'): 7, ('V', 'D'): 4
}

# Create model
model = gp.Model("WorkerAssignment")

# Add decision variables
x = model.addVars(workers, tasks, vtype=GRB.BINARY, name="x")

# Set objective function: minimize total working hours
model.setObjective(gp.quicksum(cost[i, j] * x[i, j] for i in workers for j in tasks), GRB.MINIMIZE)

# Add constraints: each task must be assigned to exactly one worker
for j in tasks:
    model.addConstr(gp.quicksum(x[i, j] for i in workers) == 1, name=f"Task_{j}")

# Add constraints: each worker can be assigned at most one task
for i in workers:
    model.addConstr(gp.quicksum(x[i, j] for j in tasks) <= 1, name=f"Worker_{i}")

# Solve the model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Minimum total working hours: {model.ObjVal}")
    print("Assignment plan:")
    for i in workers:
        for j in tasks:
            if x[i, j].X > 0.5:
                print(f"Worker {i} -> Task {j}")
else:
    print("No optimal solution found")