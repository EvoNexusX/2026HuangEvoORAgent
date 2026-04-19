import gurobipy as gp
from gurobipy import GRB

def solve_integer_programming():
    """
    Solves the integer linear programming problem:
    Minimize: 2x + 3y
    Subject to: x + y >= 10, x >= 0, y >= 0, x and y integers
    """
    
    try:
        # Create model with descriptive name
        model = gp.Model("IntegerLP_Example")
        
        # Create decision variables
        # Lower bound = 0 (non-negative), integer type
        x = model.addVar(lb=0.0, vtype=GRB.INTEGER, name="x")
        y = model.addVar(lb=0.0, vtype=GRB.INTEGER, name="y")
        
        # Update model to integrate variables
        model.update()
        
        # Set objective function: minimize 2x + 3y
        model.setObjective(2*x + 3*y, GRB.MINIMIZE)
        
        # Add constraints
        # Constraint 1: x + y >= 10
        model.addConstr(x + y >= 10, name="sum_constraint")
        
        # Optional: Set solver parameters for better performance
        model.setParam('OutputFlag', 1)      # Enable solver output
        model.setParam('MIPGap', 0.0001)     # 0.01% optimality tolerance
        
        # Solve the model
        model.optimize()
        
        # Check optimization status and display results
        if model.status == GRB.OPTIMAL:
            print("\n[SUCCESS] Optimal solution found!")
            print(f"Optimal objective value: {model.objVal:.2f}")
            print(f"x = {x.X:.0f}, y = {y.X:.0f}")
            print(f"Constraint x + y = {x.X + y.X:.0f} (>= 10)")
            
        elif model.status == GRB.INFEASIBLE:
            print("[ERROR] Model is infeasible!")
            # Compute and display the IIS (Irreducible Inconsistent Subsystem)
            model.computeIIS()
            model.write("model.ilp")
            print("Infeasible constraints written to 'model.ilp'")
            
        elif model.status == GRB.UNBOUNDED:
            print("[WARNING] Model is unbounded!")
            
        elif model.status == GRB.TIME_LIMIT:
            print("[INFO] Time limit reached!")
            if model.SolCount > 0:
                print(f"Best found objective: {model.objVal:.2f}")
                print(f"x = {x.X:.0f}, y = {y.X:.0f}")
                
        else:
            print(f"Optimization ended with status: {model.status}")
            
    except gp.GurobiError as e:
        print(f"Gurobi error occurred: {e}")
        
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

# Main execution
if __name__ == "__main__":
    solve_integer_programming()