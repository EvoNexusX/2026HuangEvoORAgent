import gurobipy as gp
from gurobipy import GRB

# Step 1: Define data
# Workers: 1 to 5 (corresponding to I, II, III, IV, V)
I = list(range(1, 6))  # [1,2,3,4,5]
# Tasks: A, B, C, D
J = ['A', 'B', 'C', 'D']

# Cost matrix: time required for each worker-task combination
# Rows: workers 1 to 5, Columns: tasks A to D
cost = {
    1: {'A': 9, 'B': 4, 'C': 3, 'D': 7},   # Worker I
    2: {'A': 4, 'B': 6, 'C': 5, 'D': 6},   # Worker II
    3: {'A': 5, 'B': 4, 'C': 7, 'D': 5},   # Worker III
    4: {'A': 7, 'B': 5, 'C': 2, 'D': 3},   # Worker IV
    5: {'A': 10, 'B': 6, 'C': 7, 'D': 4}   # Worker V
}

# Step 2: Create model
model = gp.Model('WorkerAssignment')

# Step 3: Add binary decision variables
x = model.addVars(I, J, vtype=GRB.BINARY, name='x')

# Step 4: Set objective - minimize total working hours
model.setObjective(
    gp.quicksum(cost[i][j] * x[i, j] for i in I for j in J),
    GRB.MINIMIZE
)

# Step 5: Add constraints

# Constraint 1: Each task must be assigned to exactly one worker
for j in J:
    model.addConstr(
        gp.quicksum(x[i, j] for i in I) == 1,
        name=f'task_{j}'
    )

# Constraint 2: Each worker can be assigned to at most one task
for i in I:
    model.addConstr(
        gp.quicksum(x[i, j] for j in J) <= 1,
        name=f'worker_{i}'
    )

# Step 6: Solve the model
model.optimize()

# Step 7: Output results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Minimum total working hours: {model.objVal:.0f} hours")
    print("\nAssignment plan:")
    
    assigned_workers = []
    for j in J:
        for i in I:
            if x[i, j].X > 0.5:  # Variable is effectively 1
                print(f"  Task {j} -> Worker {i} (takes {cost[i][j]} hours)")
                assigned_workers.append(i)
                break
    
    # Find which worker is not assigned
    all_workers = set(I)
    assigned_set = set(assigned_workers)
    unassigned_workers = all_workers - assigned_set
    print(f"\nUnassigned worker(s): {sorted(unassigned_workers)}")
    
elif model.status == GRB.INFEASIBLE:
    print("Model is infeasible!")
elif model.status == GRB.UNBOUNDED:
    print("Model is unbounded!")
else:
    print(f"Optimization ended with status: {model.status}")