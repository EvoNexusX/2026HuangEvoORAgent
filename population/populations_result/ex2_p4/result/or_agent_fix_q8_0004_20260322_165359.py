# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

# Define sets and parameters
workers = [1, 2, 3, 4, 5]  # Workers I, II, III, IV, V
tasks = ['A', 'B', 'C', 'D']

# Cost parameters c_ij from Table 5-2
cost = {
    (1, 'A'): 9, (1, 'B'): 4, (1, 'C'): 3, (1, 'D'): 7,
    (2, 'A'): 4, (2, 'B'): 6, (2, 'C'): 5, (2, 'D'): 6,
    (3, 'A'): 5, (3, 'B'): 4, (3, 'C'): 7, (3, 'D'): 5,
    (4, 'A'): 7, (4, 'B'): 5, (4, 'C'): 2, (4, 'D'): 3,
    (5, 'A'): 10, (5, 'B'): 6, (5, 'C'): 7, (5, 'D'): 4
}

# Create model
model = gp.Model("WorkerTaskAssignment")

# Define decision variables
x = model.addVars(workers, tasks, vtype=GRB.BINARY, name="x")

# Set objective function: minimize total working hours
model.setObjective(
    gp.quicksum(cost[i, j] * x[i, j] for i in workers for j in tasks),
    GRB.MINIMIZE
)

# Add constraints
# Each task must be assigned to exactly one worker
for j in tasks:
    model.addConstr(
        gp.quicksum(x[i, j] for i in workers) == 1,
        name=f"Task_{j}_assigned"
    )

# Each worker can be assigned at most one task
for i in workers:
    model.addConstr(
        gp.quicksum(x[i, j] for j in tasks) <= 1,
        name=f"Worker_{i}_limit"
    )

# Solve the model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Optimal total working hours: {model.objVal} hours")
    print("Assignment plan:")
    for i in workers:
        for j in tasks:
            if x[i, j].x > 0.5:
                print(f"  Worker {i} -> Task {j} (hours: {cost[i, j]})")
else:
    print("No optimal solution found")