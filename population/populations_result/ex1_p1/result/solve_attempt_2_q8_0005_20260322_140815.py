import gurobipy as gp
from gurobipy import GRB

# Step 1: Initialize the Gurobi model
model = gp.Model("Assignment_Problem")

# Step 2: Define data
workers = ['I', 'II', 'III', 'IV', 'V']
tasks = ['A', 'B', 'C', 'D']

# Cost matrix: time required for each worker-task pair
cost = {
    ('I', 'A'): 9, ('I', 'B'): 4, ('I', 'C'): 3, ('I', 'D'): 7,
    ('II', 'A'): 4, ('II', 'B'): 6, ('II', 'C'): 5, ('II', 'D'): 6,
    ('III', 'A'): 5, ('III', 'B'): 4, ('III', 'C'): 7, ('III', 'D'): 5,
    ('IV', 'A'): 7, ('IV', 'B'): 5, ('IV', 'C'): 2, ('IV', 'D'): 3,
    ('V', 'A'): 10, ('V', 'B'): 6, ('V', 'C'): 7, ('V', 'D'): 4
}

# Step 3: Create binary decision variables
x = model.addVars(workers, tasks, vtype=GRB.BINARY, name="x")

# Step 4: Set objective function - minimize total working hours
model.setObjective(
    gp.quicksum(cost[i, j] * x[i, j] for i in workers for j in tasks),
    GRB.MINIMIZE
)

# Step 5: Add constraints
# Each task must be assigned to exactly one worker
for j in tasks:
    model.addConstr(
        gp.quicksum(x[i, j] for i in workers) == 1,
        name=f"Task_{j}_assigned"
    )

# Each worker can be assigned to at most one task
for i in workers:
    model.addConstr(
        gp.quicksum(x[i, j] for j in tasks) <= 1,
        name=f"Worker_{i}_capacity"
    )

# Step 6: Optimize the model
model.optimize()

# Step 7: Output results
if model.status == GRB.OPTIMAL:
    print(f"Optimal total working hours: {model.objVal:.0f}")
    print("\nOptimal assignments:")
    
    assigned_workers = []
    for i in workers:
        for j in tasks:
            if x[i, j].X > 0.5:  # Check if variable is 1 (with tolerance)
                print(f"  Worker {i} -> Task {j} (Time: {cost[i, j]} hours)")
                assigned_workers.append(i)
    
    # Identify unassigned worker
    unassigned = set(workers) - set(assigned_workers)
    if unassigned:
        print(f"\nUnassigned worker: {unassigned.pop()}")
    else:
        print("\nAll workers are assigned (this should not happen with 5 workers and 4 tasks).")

elif model.status == GRB.INFEASIBLE:
    print("Model is infeasible.")
elif model.status == GRB.UNBOUNDED:
    print("Model is unbounded.")
else:
    print(f"Optimization ended with status {model.status}")