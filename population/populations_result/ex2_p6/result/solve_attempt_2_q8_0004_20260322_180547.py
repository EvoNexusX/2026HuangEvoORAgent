#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Worker Assignment Problem
Minimize total working hours by assigning 4 tasks to 4 out of 5 workers
"""

import gurobipy as gp
from gurobipy import GRB

def solve_assignment_problem():
    """
    Solves the worker assignment problem to minimize total working hours.
    
    Problem: Assign 4 tasks (A, B, C, D) to 4 out of 5 workers (I, II, III, IV, V)
    such that each task is assigned to exactly one worker and each worker
    is assigned to at most one task.
    """
    
    # ======================
    # 1. DATA DEFINITION
    # ======================
    
    # Worker and task identifiers
    workers = ['I', 'II', 'III', 'IV', 'V']
    tasks = ['A', 'B', 'C', 'D']
    
    # Cost matrix (working hours): worker -> task -> hours
    # Based on Table 5-2
    cost_data = {
        'I': {'A': 9, 'B': 4, 'C': 3, 'D': 7},
        'II': {'A': 4, 'B': 6, 'C': 5, 'D': 6},
        'III': {'A': 5, 'B': 4, 'C': 7, 'D': 5},
        'IV': {'A': 7, 'B': 5, 'C': 2, 'D': 3},
        'V': {'A': 10, 'B': 6, 'C': 7, 'D': 4}
    }
    
    try:
        # ======================
        # 2. MODEL CREATION
        # ======================
        
        # Create a new model
        model = gp.Model("WorkerAssignment")
        model.ModelSense = GRB.MINIMIZE  # Minimize objective
        
        # Optional: Set solver parameters
        model.Params.OutputFlag = 1  # Enable solver output
        model.Params.LogToConsole = 1  # Log to console
        model.Params.TimeLimit = 30  # Time limit in seconds
        model.Params.MIPGap = 0.0001  # Optimality gap tolerance
        
        # ======================
        # 3. DECISION VARIABLES
        # ======================
        
        print("Creating decision variables...")
        # Binary variables: x[worker, task] = 1 if worker assigned to task
        x = {}
        for worker in workers:
            for task in tasks:
                var_name = f"x_{worker}_{task}"
                x[(worker, task)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=var_name
                )
        
        model.update()  # Incorporate variables into model
        
        # ======================
        # 4. OBJECTIVE FUNCTION
        # ======================
        
        print("Setting objective function...")
        # Minimize total working hours
        objective_expr = gp.quicksum(
            cost_data[worker][task] * x[(worker, task)]
            for worker in workers
            for task in tasks
        )
        model.setObjective(objective_expr)
        
        # ======================
        # 5. CONSTRAINTS
        # ======================
        
        print("Adding constraints...")
        
        # Constraint 1: Each task assigned to exactly one worker
        for task in tasks:
            task_expr = gp.quicksum(x[(worker, task)] for worker in workers)
            model.addConstr(task_expr == 1, name=f"Task_{task}_assigned")
        
        # Constraint 2: Each worker assigned to at most one task
        for worker in workers:
            worker_expr = gp.quicksum(x[(worker, task)] for task in tasks)
            model.addConstr(worker_expr <= 1, name=f"Worker_{worker}_limit")
        
        model.update()  # Incorporate constraints into model
        
        # ======================
        # 6. SOLVE THE MODEL
        # ======================
        
        print("\nSolving the optimization model...")
        model.optimize()
        
        # ======================
        # 7. SOLUTION ANALYSIS
        # ======================
        
        print("\n" + "="*60)
        
        # Check optimization status
        status = model.status
        
        if status == GRB.OPTIMAL:
            print("OPTIMAL SOLUTION FOUND")
            print("="*60)
            
            # Extract and display solution
            total_hours = model.ObjVal
            print(f"\nOptimal total working hours: {total_hours:.0f}")
            
            print("\nAssignment details:")
            print("-"*40)
            
            assigned_count = 0
            unassigned_worker = None
            
            for worker in workers:
                assigned = False
                for task in tasks:
                    if x[(worker, task)].X > 0.5:  # Variable value close to 1
                        hours = cost_data[worker][task]
                        print(f"Worker {worker:4s} -> Task {task}: {hours} hours")
                        assigned = True
                        assigned_count += 1
                
                if not assigned:
                    unassigned_worker = worker
                    print(f"Worker {worker:4s} -> Not assigned")
            
            print("-"*40)
            print(f"Total assignments: {assigned_count} out of {len(tasks)} tasks")
            print(f"Unassigned worker: {unassigned_worker}")
            
            # Verify solution consistency
            print("\nSolution verification:")
            print("-"*40)
            
            # Check task assignments
            for task in tasks:
                assigned_workers = [w for w in workers if x[(w, task)].X > 0.5]
                print(f"Task {task}: Assigned to worker(s) {assigned_workers}")
            
            # Check worker assignments
            for worker in workers:
                assigned_tasks = [t for t in tasks if x[(worker, t)].X > 0.5]
                assignment_count = len(assigned_tasks)
                print(f"Worker {worker}: Assigned to {assignment_count} task(s) {assigned_tasks}")
            
            print("="*60)
            
        elif status == GRB.INFEASIBLE:
            print("MODEL IS INFEASIBLE")
            print("="*60)
            print("\nNo assignment satisfies all constraints.")
            
            # Compute Irreducible Inconsistent Subsystem (IIS) for debugging
            print("\nComputing IIS to identify conflicting constraints...")
            model.computeIIS()
            model.write("assignment_model.ilp")
            print("IIS written to file 'assignment_model.ilp'")
            
        elif status == GRB.UNBOUNDED:
            print("MODEL IS UNBOUNDED")
            print("="*60)
            print("\nObjective can be improved indefinitely.")
            
        elif status == GRB.TIME_LIMIT:
            print("TIME LIMIT REACHED")
            print("="*60)
            if model.SolCount > 0:
                print(f"\nBest solution found: {model.ObjVal:.2f} hours")
                print("Solution may not be optimal.")
            else:
                print("\nNo feasible solution found within time limit.")
                
        elif status == GRB.INTERRUPTED:
            print("OPTIMIZATION INTERRUPTED")
            print("="*60)
            if model.SolCount > 0:
                print(f"\nBest solution found: {model.ObjVal:.2f} hours")
                
        else:
            print(f"SOLVER TERMINATED WITH STATUS: {status}")
            print("="*60)
            
        # ======================
        # 8. MODEL STATISTICS
        # ======================
        
        print("\nModel Statistics:")
        print("-"*40)
        print(f"Number of variables: {model.NumVars}")
        print(f"Number of constraints: {model.NumConstrs}")
        print(f"Number of non-zero coefficients: {model.NumNZs}")
        
        if model.SolCount > 0:
            print(f"\nSolution time: {model.Runtime:.3f} seconds")
            print(f"Optimality gap: {model.MIPGap:.6f}")
            print(f"Node count: {model.NodeCount}")
            print(f"Iteration count: {model.IterCount}")
        
        print("="*60)
        
    except gp.GurobiError as e:
        print(f"\nGurobi error during optimization:")
        print(f"Error code: {e.errno}")
        print(f"Error message: {e}")
        
    except Exception as e:
        print(f"\nUnexpected error during execution:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")


# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    print("Worker Assignment Problem - Minimizing Total Working Hours")
    print("="*60)
    print("Problem: Assign 4 tasks (A, B, C, D) to 4 out of 5 workers")
    print("Goal: Minimize total working hours")
    print("="*60)
    
    solve_assignment_problem()
    
    print("\nProgram completed.")