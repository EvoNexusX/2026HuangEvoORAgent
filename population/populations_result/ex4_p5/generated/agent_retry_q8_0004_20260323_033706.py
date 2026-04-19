import gurobipy as gp
from gurobipy import GRB

# Step 1: Prepare data
workers = ['I', 'II', 'III', 'IV', 'V']
tasks = ['A', 'B', 'C', 'D']

cost = {
    ('I', 'A'): 9, ('I', 'B'): 4, ('I', 'C'): 3, ('I', 'D'): 7,
    ('II', 'A'): 4, ('II', 'B'): 6, ('II', 'C'): 5, ('II', 'D'): 6,
    ('III', 'A'): 5, ('III', 'B'): 4, ('III', 'C'): 7, ('III', 'D'): 5,
    ('IV', 'A'): 7, ('IV', 'B'): 5, ('IV', 'C'): 2, ('IV', 'D'): 3,
    ('V', 'A'): 10, ('V', 'B'): 6, ('V', 'C'): 7, ('V', 'D'): 4
}

# Step 2: Create model
model = gp.Model("Assignment_Problem")

# Step 3: Define decision variables
x = model.addVars(workers, tasks, vtype=GRB.BINARY, name="x")

# Step 4: Set objective function
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

# Each worker can do at most one task
for i in workers:
    model.addConstr(
        gp.quicksum(x[i, j] for j in tasks) <= 1,
        name=f"Worker_{i}_limit"
    )

# Step 6: Solve the model
model.optimize()

# Step 7: Output results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    print(f"Minimum total working hours: {model.ObjVal}")
    print("\nDetailed assignment plan:")
    total_hours = 0
    for i in workers:
        for j in tasks:
            if x[i, j].X > 0.5:
                print(f"Worker {i} -> Task {j} (Hours: {cost[i, j]})")
                total_hours += cost[i, j]
    print(f"\nTotal hours: {total_hours}")
else:
    print(f"No optimal solution found. Status code: {model.status}")