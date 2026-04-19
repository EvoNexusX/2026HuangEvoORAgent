# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("Assignment")

# Define worker and task sets
workers = ['I', 'II', 'III', 'IV', 'V']
tasks = ['A', 'B', 'C', 'D']

# Define work time cost data from Table 5-2
cost = {
    ('I', 'A'): 9, ('I', 'B'): 4, ('I', 'C'): 3, ('I', 'D'): 7,
    ('II', 'A'): 4, ('II', 'B'): 6, ('II', 'C'): 5, ('II', 'D'): 6,
    ('III', 'A'): 5, ('III', 'B'): 4, ('III', 'C'): 7, ('III', 'D'): 5,
    ('IV', 'A'): 7, ('IV', 'B'): 5, ('IV', 'C'): 2, ('IV', 'D'): 3,
    ('V', 'A'): 10, ('V', 'B'): 6, ('V', 'C'): 7, ('V', 'D'): 4
}

# Create decision variables
x = model.addVars(workers, tasks, vtype=GRB.BINARY, name="x")

# Set objective function: minimize total work time
model.setObjective(gp.quicksum(cost[w, t] * x[w, t] for w in workers for t in tasks), GRB.MINIMIZE)

# Add constraints: each task must be done by exactly one worker
model.addConstrs((x.sum('*', t) == 1 for t in tasks), name="TaskAssignment")

# Add constraints: each worker can do at most one task
model.addConstrs((x.sum(w, '*') <= 1 for w in workers), name="WorkerLimit")

# Solve model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Minimum total work time: {model.ObjVal:.0f}")
    print("Assignment plan:")
    for w in workers:
        for t in tasks:
            if x[w, t].X > 0.5:
                print(f"  Worker {w} -> Task {t}")
else:
    print(f"Optimization failed, status: {model.status}")