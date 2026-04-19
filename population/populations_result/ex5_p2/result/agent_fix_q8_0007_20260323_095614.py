import gurobipy as gp
from gurobipy import GRB

def solve_assignment():
    # Define data
    I = ["I", "II", "III", "IV", "V"]  # Worker set
    J = ["A", "B", "C", "D"]           # Task set
    
    # Cost data (working hours)
    c = {
        ("I", "A"): 9, ("I", "B"): 4, ("I", "C"): 3, ("I", "D"): 7,
        ("II", "A"): 4, ("II", "B"): 6, ("II", "C"): 5, ("II", "D"): 6,
        ("III", "A"): 5, ("III", "B"): 4, ("III", "C"): 7, ("III", "D"): 5,
        ("IV", "A"): 7, ("IV", "B"): 5, ("IV", "C"): 2, ("IV", "D"): 3,
        ("V", "A"): 10, ("V", "B"): 6, ("V", "C"): 7, ("V", "D"): 4
    }
    
    # Create model
    m = gp.Model("assignment")
    
    # Add variables
    x = m.addVars(I, J, vtype=GRB.BINARY, name="x")
    
    # Set objective function
    m.setObjective(gp.quicksum(c[i, j] * x[i, j] for i in I for j in J), GRB.MINIMIZE)
    
    # Add constraints: each task is done by exactly one worker
    m.addConstrs((x.sum('*', j) == 1 for j in J), name="task")
    
    # Add constraints: each worker does at most one task
    m.addConstrs((x.sum(i, '*') <= 1 for i in I), name="worker")
    
    # Solve model
    m.optimize()
    
    # Output results
    if m.status == GRB.OPTIMAL:
        print("Optimal total time:", m.objVal)
        print("\nAssignment plan:")
        for i in I:
            for j in J:
                if x[i, j].X > 0.5:
                    print(f"Worker {i} -> Task {j} (Hours: {c[i, j]})")
    else:
        print("No optimal solution found")
        print("Solution status:", m.status)

if __name__ == "__main__":
    solve_assignment()