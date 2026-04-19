import gurobipy as gp
from gurobipy import GRB

def solve_assignment_problem():
    """
    Solves the worker-task assignment problem to minimize total working hours.
    
    Problem: Assign 4 out of 5 workers to 4 different tasks such that each task
    is done by exactly one worker, each worker does at most one task, and total
    working hours are minimized.
    """
    try:
        # ====================
        # 1. Model Setup
        # ====================
        model = gp.Model("WorkerAssignment")
        model.setParam('OutputFlag', 1)  # Enable solver output
        model.setParam('TimeLimit', 30)  # Optional time limit
        
        # ====================
        # 2. Data Definition
        # ====================
        workers = ['I', 'II', 'III', 'IV', 'V']
        tasks = ['A', 'B', 'C', 'D']
        
        # Cost matrix (hours required) from Table 5-2
        # Structure: cost[(worker, task)] = hours
        cost_matrix = {
            ('I', 'A'): 9, ('I', 'B'): 4, ('I', 'C'): 3, ('I', 'D'): 7,
            ('II', 'A'): 4, ('II', 'B'): 6, ('II', 'C'): 5, ('II', 'D'): 6,
            ('III', 'A'): 5, ('III', 'B'): 4, ('III', 'C'): 7, ('III', 'D'): 5,
            ('IV', 'A'): 7, ('IV', 'B'): 5, ('IV', 'C'): 2, ('IV', 'D'): 3,
            ('V', 'A'): 10, ('V', 'B'): 6, ('V', 'C'): 7, ('V', 'D'): 4,
        }
        
        # ====================
        # 3. Data Validation
        # ====================
        if len(cost_matrix) != len(workers) * len(tasks):
            raise ValueError("Cost matrix size doesn't match workers × tasks")
        
        # Check for non-negative costs (assignment problem requirement)
        for hours in cost_matrix.values():
            if hours < 0:
                raise ValueError("All working hours must be non-negative")
        
        # ====================
        # 4. Variable Creation
        # ====================
        # Binary decision variables: x[worker, task] = 1 if assigned, 0 otherwise
        x = model.addVars(workers, tasks, vtype=GRB.BINARY, name="assign")
        
        # ====================
        # 5. Objective Function
        # ====================
        # Minimize total working hours: sum(cost[worker,task] * x[worker,task])
        objective = gp.quicksum(cost_matrix[(i, j)] * x[i, j] 
                               for i in workers for j in tasks)
        model.setObjective(objective, GRB.MINIMIZE)
        
        # ====================
        # 6. Constraints
        # ====================
        # Constraint 1: Each task must be assigned to exactly one worker
        for task in tasks:
            model.addConstr(gp.quicksum(x[worker, task] for worker in workers) == 1,
                           name=f"Task_{task}_assigned")
        
        # Constraint 2: Each worker can do at most one task (since 5 workers, 4 tasks)
        for worker in workers:
            model.addConstr(gp.quicksum(x[worker, task] for task in tasks) <= 1,
                           name=f"Worker_{worker}_limit")
        
        # ====================
        # 7. Model Optimization
        # ====================
        print("Solving assignment problem...")
        model.optimize()
        
        # ====================
        # 8. Result Processing
        # ====================
        status = model.status
        
        if status == GRB.OPTIMAL:
            # Optimal solution found
            print("\n" + "="*50)
            print("OPTIMAL SOLUTION FOUND")
            print("="*50)
            print(f"Minimum total working hours: {model.objVal:.0f}")
            print("\nOptimal assignment plan:")
            print("-" * 30)
            
            assignments = []
            total_hours = 0
            
            # Collect and sort assignments by task for better readability
            for worker in workers:
                for task in tasks:
                    if x[worker, task].X > 0.5:  # Check if variable is 1 (with tolerance)
                        hours = cost_matrix[(worker, task)]
                        assignments.append((task, worker, hours))
                        total_hours += hours
            
            # Sort by task for organized output
            assignments.sort(key=lambda a: a[0])
            
            for task, worker, hours in assignments:
                print(f"  Task {task} → Worker {worker} (hours: {hours})")
            
            # Find unassigned worker(s)
            assigned_workers = {worker for _, worker, _ in assignments}
            unassigned = set(workers) - assigned_workers
            if unassigned:
                print(f"\nUnassigned worker(s): {', '.join(sorted(unassigned))}")
            
            print("-" * 30)
            print(f"Total hours verification: {total_hours}")
            
        elif status == GRB.INFEASIBLE:
            print("\nERROR: Model is infeasible.")
            print("Computing Irreducible Inconsistent Subsystem (IIS)...")
            model.computeIIS()
            model.write("assignment_problem_iis.ilp")
            print("IIS written to file 'assignment_problem_iis.ilp'")
            print("Review the file to identify conflicting constraints.")
            
        elif status == GRB.UNBOUNDED:
            print("\nERROR: Model is unbounded.")
            print("This shouldn't happen in an assignment problem. Check objective coefficients.")
            
        elif status == GRB.TIME_LIMIT:
            print(f"\nWARNING: Time limit reached. Best solution: {model.objVal:.2f}")
            if model.SolCount > 0:
                print("Displaying best found solution:")
                for worker in workers:
                    for task in tasks:
                        if x[worker, task].X > 0.5:
                            hours = cost_matrix[(worker, task)]
                            print(f"  Task {task} → Worker {worker} (hours: {hours})")
            
        else:
            print(f"\nOptimization terminated with status: {status}")
            print("No optimal solution available.")
    
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
    except ValueError as e:
        print(f"Data error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

# ====================
# 9. Main Execution
# ====================
if __name__ == "__main__":
    solve_assignment_problem()