from gurobipy import Model, GRB, quicksum

# Step 1: Initialize the model
model = Model('assignment')

# Step 2: Define sets and parameters
workers = [1, 2, 3, 4, 5]
tasks = ['A', 'B', 'C', 'D']

time = {
    (1, 'A'): 9, (1, 'B'): 4, (1, 'C'): 3, (1, 'D'): 7,
    (2, 'A'): 4, (2, 'B'): 6, (2, 'C'): 5, (2, 'D'): 6,
    (3, 'A'): 5, (3, 'B'): 4, (3, 'C'): 7, (3, 'D'): 5,
    (4, 'A'): 7, (4, 'B'): 5, (4, 'C'): 2, (4, 'D'): 3,
    (5, 'A'): 10, (5, 'B'): 6, (5, 'C'): 7, (5, 'D'): 4
}

# Step 3: Define decision variables
x = model.addVars(workers, tasks, vtype=GRB.BINARY, name='x')

# Step 4: Set objective function
model.setObjective(
    quicksum(time[i, j] * x[i, j] for i in workers for j in tasks),
    GRB.MINIMIZE
)

# Step 5: Add constraints
# Each task must be completed by exactly one worker
for j in tasks:
    model.addConstr(
        quicksum(x[i, j] for i in workers) == 1,
        name=f'task_{j}'
    )

# Each worker can do at most one task
for i in workers:
    model.addConstr(
        quicksum(x[i, j] for j in tasks) <= 1,
        name=f'worker_{i}'
    )

# Step 6: Solve the model
model.optimize()

# Step 7: Output results
if model.status == GRB.OPTIMAL:
    print(f'Optimal total working hours: {model.objVal} hours')
    print('\nAssignment plan:')
    for i in workers:
        assigned = False
        for j in tasks:
            if x[i, j].x > 0.5:
                print(f'Worker {i} -> Task {j} (time: {time[i, j]} hours)')
                assigned = True
        if not assigned:
            print(f'Worker {i} -> No task assigned')
    print('\nTask assignment details:')
    for j in tasks:
        for i in workers:
            if x[i, j].x > 0.5:
                print(f'Task {j} -> Worker {i}')
else:
    print('No optimal solution found')