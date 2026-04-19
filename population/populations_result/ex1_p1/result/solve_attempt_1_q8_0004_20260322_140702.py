import gurobipy as gp
from gurobipy import GRB

def solve_assignment():
    """
    Solve the worker-task assignment problem to minimize total working hours.
    There are 5 workers and 4 tasks, so 1 worker will remain unassigned.
    """
    # Create Gurobi model
    model = gp.Model("WorkerAssignment")
    
    # Define workers and tasks
    workers = ['I', 'II', 'III', 'IV', 'V']
    tasks = ['A', 'B', 'C', 'D']
    
    # Cost matrix: hours required for worker i to complete task j
    cost = {
        'I': {'A': 9, 'B': 4, 'C': 3, 'D': 7},
        'II': {'A': 4, 'B': 6, 'C': 5, 'D': 6},
        'III': {'A': 5, 'B': 4, 'C': 7, 'D': 5},
        'IV': {'A': 7, 'B': 5, 'C': 2, 'D': 3},
        'V': {'A': 10, 'B': 6, 'C': 7, 'D': 4}
    }
    
    # Create binary decision variables: x[i,j] = 1 if worker i assigned to task j
    x = model.addVars(workers, tasks, vtype=GRB.BINARY, name="assign")
    
    # Set objective: minimize total working hours
    model.setObjective(
        gp.quicksum(cost[i][j] * x[i, j] for i in workers for j in tasks),
        GRB.MINIMIZE
    )
    
    # Constraints: each task must be assigned to exactly one worker
    for j in tasks:
        model.addConstr(
            gp.quicksum(x[i, j] for i in workers) == 1,
            name=f"task_{j}"
        )
    
    # Constraints: each worker can do at most one task
    for i in workers:
        model.addConstr(
            gp.quicksum(x[i, j] for j in tasks) <= 1,
            name=f"worker_{i}"
        )
    
    # Optimize the model
    model.optimize()
    
    # Check and display results
    if model.status == GRB.OPTIMAL:
        print(f"Optimal total working hours: {model.ObjVal:.0f}\n")
        print("Optimal assignment:")
        
        assigned_count = 0
        for i in workers:
            worker_assigned = False
            for j in tasks:
                if x[i, j].X > 0.5:  # Binary variable is 1
                    print(f"  Worker {i} → Task {j} (hours: {cost[i][j]})")
                    worker_assigned = True
                    assigned_count += 1
            if not worker_assigned:
                print(f"  Worker {i} → Not assigned")
        
        print(f"\nTotal assigned workers: {assigned_count}")
    elif model.status == GRB.INFEASIBLE:
        print("Model is infeasible")
    elif model.status == GRB.UNBOUNDED:
        print("Model is unbounded")
    else:
        print(f"Optimization ended with status {model.status}")

if __name__ == "__main__":
    solve_assignment()