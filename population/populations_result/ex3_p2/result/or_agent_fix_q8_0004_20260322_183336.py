# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

# Initialize model
model = gp.Model("Worker_Task_Assignment")

# Worker set
workers = ['I', 'II', 'III', 'IV', 'V']

# Task set
tasks = ['A', 'B', 'C', 'D']

# Work hour matrix (workers × tasks)
cost_matrix = {
    'I': {'A': 9, 'B': 4, 'C': 3, 'D': 7},
    'II': {'A': 4, 'B': 6, 'C': 5, 'D': 6},
    'III': {'A': 5, 'B': 4, 'C': 7, 'D': 5},
    'IV': {'A': 7, 'B': 5, 'C': 2, 'D': 3},
    'V': {'A': 10, 'B': 6, 'C': 7, 'D': 4}
}

# Create binary decision variables
x = model.addVars(
    workers, 
    tasks, 
    vtype=GRB.BINARY,
    name="x"
)

# Set objective: minimize total working hours
model.setObjective(
    gp.quicksum(
        cost_matrix[i][j] * x[i, j] 
        for i in workers 
        for j in tasks
    ),
    GRB.MINIMIZE
)

# Add constraints: each task must be assigned to exactly one worker
for j in tasks:
    model.addConstr(
        gp.quicksum(x[i, j] for i in workers) == 1,
        name=f"Task_{j}_Coverage"
    )

# Add constraints: each worker can be assigned to at most one task
for i in workers:
    model.addConstr(
        gp.quicksum(x[i, j] for j in tasks) <= 1,
        name=f"Worker_{i}_Capacity"
    )

# Solve the model
model.optimize()

# Check solution status and output results
if model.status == GRB.OPTIMAL:
    print(f"Optimal objective value (total working hours): {model.objVal}")
    print("\nAssignment plan:")
    assignment = {}
    for i in workers:
        for j in tasks:
            if x[i, j].X > 0.5:  # Check if variable equals 1
                assignment[j] = i
                print(f"  Task {j} → Worker {i}")
    
    # Identify unassigned worker
    assigned_workers = set(assignment.values())
    unassigned = [i for i in workers if i not in assigned_workers]
    if unassigned:
        print(f"\nUnassigned worker: {', '.join(unassigned)}")
else:
    print("No optimal solution found")