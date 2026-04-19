import gurobipy as gp
from gurobipy import GRB

def solve_worker_assignment():
    try:
        # Create model
        model = gp.Model("WorkerAssignment")
        
        # Data
        workers = [1, 2, 3, 4, 5]  # I, II, III, IV, V
        tasks = ['A', 'B', 'C', 'D']
        
        # Cost matrix: workers x tasks
        cost = {
            1: {'A': 9, 'B': 4, 'C': 3, 'D': 7},
            2: {'A': 4, 'B': 6, 'C': 5, 'D': 6},
            3: {'A': 5, 'B': 4, 'C': 7, 'D': 5},
            4: {'A': 7, 'B': 5, 'C': 2, 'D': 3},
            5: {'A': 10, 'B': 6, 'C': 7, 'D': 4}
        }
        
        # Create binary variables
        x = model.addVars(
            workers, 
            tasks, 
            vtype=GRB.BINARY, 
            name="x"
        )
        
        # Set objective: minimize total working hours
        model.setObjective(
            gp.quicksum(
                cost[i][j] * x[i, j] 
                for i in workers 
                for j in tasks
            ), 
            GRB.MINIMIZE
        )
        
        # Constraints
        
        # Each task must be assigned to exactly one worker
        for j in tasks:
            model.addConstr(
                gp.quicksum(x[i, j] for i in workers) == 1,
                name=f"task_{j}_assigned"
            )
        
        # Each worker can do at most one task
        for i in workers:
            model.addConstr(
                gp.quicksum(x[i, j] for j in tasks) <= 1,
                name=f"worker_{i}_limit"
            )
        
        # Optimize model
        model.optimize()
        
        # Check solution status
        if model.status == GRB.OPTIMAL:
            print(f"Optimal solution found with total hours: {model.objVal:.0f}\n")
            
            # Print assignment details
            print("Optimal Assignment:")
            assigned_workers = []
            for j in tasks:
                for i in workers:
                    if x[i, j].x > 0.5:
                        worker_name = ['I', 'II', 'III', 'IV', 'V'][i-1]
                        print(f"  Task {j} → Worker {worker_name} (Hours: {cost[i][j]})")
                        assigned_workers.append(i)
            
            # Find unassigned worker
            for i in workers:
                if i not in assigned_workers:
                    worker_name = ['I', 'II', 'III', 'IV', 'V'][i-1]
                    print(f"  Worker {worker_name} is unassigned")
                    
        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible")
            # Compute IIS to identify conflicting constraints
            model.computeIIS()
            print("Conflicting constraints:")
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"  {c.constrName}")
                    
        elif model.status == GRB.UNBOUNDED:
            print("Model is unbounded")
            
        elif model.status == GRB.INF_OR_UNBD:
            print("Model is infeasible or unbounded")
            
        else:
            print(f"No optimal solution found. Status code: {model.status}")
            
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        
    except Exception as e:
        print(f"General error: {e}")
        
    finally:
        # Properly dispose of model
        if 'model' in locals():
            model.dispose()

if __name__ == "__main__":
    solve_worker_assignment()