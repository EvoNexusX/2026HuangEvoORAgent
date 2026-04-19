import gurobipy as gp
from gurobipy import GRB

def solve_linear_program():
    """
    Solves the linear programming problem:
    Maximize: 3x + 4y
    Subject to:
        x + 2y <= 10
        2x + y <= 10
        x >= 0
        y >= 0
    """
    try:
        # 1. Create model with appropriate name
        model = gp.Model("SimpleLP")
        
        # Optional: Enable solver output for progress tracking
        model.setParam('OutputFlag', 1)
        
        # 2. Create variables with correct types and bounds
        x = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x")
        y = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y")
        
        # Update model to incorporate variables
        model.update()
        
        # 3. Set objective function (maximize)
        objective = 3*x + 4*y
        model.setObjective(objective, GRB.MAXIMIZE)
        
        # 4. Add all constraints
        model.addConstr(x + 2*y <= 10, "c1")
        model.addConstr(2*x + y <= 10, "c2")
        
        # 5. Optimize the model
        model.optimize()
        
        # 6. Check optimization status and display results
        if model.status == GRB.OPTIMAL:
            print(f"\nOptimal solution found!")
            print(f"Objective value (maximized): {model.objVal:.6f}")
            print(f"x = {x.x:.6f}")
            print(f"y = {y.x:.6f}")
            
            # Display constraint slacks
            print("\nConstraint analysis:")
            for constr in model.getConstrs():
                slack = constr.getAttr('Slack')
                print(f"  {constr.getAttr('ConstrName')}: Slack = {slack:.6f}")
                
        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible!")
            # Compute and display the Irreducible Inconsistent Subsystem (IIS)
            model.computeIIS()
            model.write("model.ilp")
            print("IIS written to file 'model.ilp'")
            
        elif model.status == GRB.UNBOUNDED:
            print("Model is unbounded!")
            
        elif model.status == GRB.INF_OR_UNBD:
            print("Model is infeasible or unbounded!")
            
        else:
            print(f"Optimization ended with status code: {model.status}")
            
        return model
        
    except gp.GurobiError as e:
        print(f"Gurobi error occurred: {e}")
        return None
    except Exception as e:
        print(f"General error occurred: {e}")
        return None

# Execute the function to solve the linear program
if __name__ == "__main__":
    model = solve_linear_program()
    
    # Optional: Save model to file for inspection
    if model is not None:
        model.write("simple_lp.lp")
        print("\nModel saved to 'simple_lp.lp' for inspection.")