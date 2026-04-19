# -*- coding: utf-8 -*-
from gurobipy import *

try:
    # Define workers and tasks
    workers = ['I', 'II', 'III', 'IV', 'V']
    tasks = ['A', 'B', 'C', 'D']
    
    # Define time required for each worker-task combination
    cost = {
        ('I', 'A'): 9, ('I', 'B'): 4, ('I', 'C'): 3, ('I', 'D'): 7,
        ('II', 'A'): 4, ('II', 'B'): 6, ('II', 'C'): 5, ('II', 'D'): 6,
        ('III', 'A'): 5, ('III', 'B'): 4, ('III', 'C'): 7, ('III', 'D'): 5,
        ('IV', 'A'): 7, ('IV', 'B'): 5, ('IV', 'C'): 2, ('IV', 'D'): 3,
        ('V', 'A'): 10, ('V', 'B'): 6, ('V', 'C'): 7, ('V', 'D'): 4
    }
    
    # Create model
    model = Model("Assignment_Problem")
    
    # Create decision variables
    x = {}
    for i in workers:
        for j in tasks:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # Set objective function: minimize total working hours
    model.setObjective(quicksum(cost[i, j] * x[i, j] for i in workers for j in tasks), GRB.MINIMIZE)
    
    # Add constraints: each task must be assigned to exactly one worker
    for j in tasks:
        model.addConstr(quicksum(x[i, j] for i in workers) == 1, name=f"task_{j}")
    
    # Add constraints: each worker can be assigned at most one task
    for i in workers:
        model.addConstr(quicksum(x[i, j] for j in tasks) <= 1, name=f"worker_{i}")
    
    # Solve the model
    model.optimize()
    
    # Output results
    if model.status == GRB.OPTIMAL:
        print(f"Optimal total working hours: {model.objVal} hours")
        print("Assignment plan:")
        for i in workers:
            for j in tasks:
                if x[i, j].X > 0.5:
                    print(f"  Worker {i} -> Task {j}, Time: {cost[i, j]} hours")
        # Output unselected workers
        unused = [i for i in workers if all(x[i, j].X < 0.5 for j in tasks)]
        if unused:
            print(f"Unselected workers: {unused}")
    else:
        print("No optimal solution found")

except GurobiError as e:
    print(f"Gurobi error: {e}")
except Exception as e:
    print(f"Other error: {e}")
finally:
    model.dispose()