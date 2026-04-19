# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("WorkerAssignment")

# Define sets
workers = ['I', 'II', 'III', 'IV', 'V']  # Worker set
tasks = ['A', 'B', 'C', 'D']             # Task set

# Time data: hours required for each worker-task combination
time_data = {
    ('I', 'A'): 9, ('I', 'B'): 4, ('I', 'C'): 3, ('I', 'D'): 7,
    ('II', 'A'): 4, ('II', 'B'): 6, ('II', 'C'): 5, ('II', 'D'): 6,
    ('III', 'A'): 5, ('III', 'B'): 4, ('III', 'C'): 7, ('III', 'D'): 5,
    ('IV', 'A'): 7, ('IV', 'B'): 5, ('IV', 'C'): 2, ('IV', 'D'): 3,
    ('V', 'A'): 10, ('V', 'B'): 6, ('V', 'C'): 7, ('V', 'D'): 4
}

# Create decision variables
assign_vars = {}
for worker in workers:
    for task in tasks:
        assign_vars[worker, task] = model.addVar(
            vtype=GRB.BINARY, 
            name=f"assign_{worker}_{task}"
        )

# Set objective: minimize total working hours
model.setObjective(
    gp.quicksum(
        time_data[worker, task] * assign_vars[worker, task] 
        for worker in workers 
        for task in tasks
    ),
    GRB.MINIMIZE
)

# Add constraints
# Each task must be assigned to exactly one worker
for task in tasks:
    model.addConstr(
        gp.quicksum(assign_vars[worker, task] for worker in workers) == 1,
        name=f"task_{task}_assignment"
    )

# Each worker can be assigned to at most one task
for worker in workers:
    model.addConstr(
        gp.quicksum(assign_vars[worker, task] for task in tasks) <= 1,
        name=f"worker_{worker}_limit"
    )

# Solve the model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Optimal total working hours: {model.ObjVal}")
    print("Assignment plan:")
    for worker in workers:
        for task in tasks:
            if assign_vars[worker, task].X > 0.5:
                print(f"Worker {worker} -> Task {task} (hours: {time_data[worker, task]})")
else:
    print("No optimal solution found")