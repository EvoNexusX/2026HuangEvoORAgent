import gurobipy as gp
from gurobipy import GRB

def solve_haus_toys_problem():
    """
    Solve the Haus Toys production planning problem using MILP.
    Returns the solution dictionary if optimal solution found, None otherwise.
    """
    try:
        # Create model with environment context
        with gp.Env(empty=True) as env:
            env.setParam('LogToConsole', 1)
            env.start()
            
            model = gp.Model("HausToysProduction", env=env)
            
            # Create decision variables
            x_t = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_t")  # trucks
            x_a = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_a")  # airplanes
            x_b = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_b")  # boats
            x_r = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_r")  # trains
            
            y_t = model.addVar(vtype=GRB.BINARY, name="y_t")
            y_a = model.addVar(vtype=GRB.BINARY, name="y_a")
            y_b = model.addVar(vtype=GRB.BINARY, name="y_b")
            y_r = model.addVar(vtype=GRB.BINARY, name="y_r")
            
            # Set objective function: maximize profit
            profit = 5*x_t + 10*x_a + 8*x_b + 7*x_r
            model.setObjective(profit, GRB.MAXIMIZE)
            
            # Resource constraints
            model.addConstr(12*x_t + 20*x_a + 15*x_b + 10*x_r <= 890, "wood_availability")
            model.addConstr(6*x_t + 3*x_a + 5*x_b + 4*x_r <= 500, "steel_availability")
            
            # Logical constraints
            model.addConstr(y_t + y_r <= 1, "truck_train_exclusive")
            model.addConstr(y_b <= y_a, "boat_implies_airplane")
            model.addConstr(x_b <= x_r, "boats_not_exceed_trains")
            
            # Linking constraints (using precomputed upper bounds)
            model.addConstr(x_t <= 74 * y_t, "link_truck_upper")
            model.addConstr(x_t >= y_t, "link_truck_lower")
            model.addConstr(x_a <= 44 * y_a, "link_airplane_upper")
            model.addConstr(x_a >= y_a, "link_airplane_lower")
            model.addConstr(x_b <= 59 * y_b, "link_boat_upper")
            model.addConstr(x_b >= y_b, "link_boat_lower")
            model.addConstr(x_r <= 89 * y_r, "link_train_upper")
            model.addConstr(x_r >= y_r, "link_train_lower")
            
            # Optimize the model
            model.optimize()
            
            # Process and display results based on solution status
            if model.status == GRB.OPTIMAL:
                print(f"Optimal solution found!")
                print(f"Maximum profit: ${model.objVal:.2f}")
                print("\nProduction quantities:")
                print(f"  Toy trucks: {x_t.x}")
                print(f"  Toy airplanes: {x_a.x}")
                print(f"  Toy boats: {x_b.x}")
                print(f"  Toy trains: {x_r.x}")
                print(f"\nIndicator variables (1 if produced, 0 otherwise):")
                print(f"  Trucks: {y_t.x}, Airplanes: {y_a.x}, Boats: {y_b.x}, Trains: {y_r.x}")
                
                # Calculate resource usage
                wood_used = 12*x_t.x + 20*x_a.x + 15*x_b.x + 10*x_r.x
                steel_used = 6*x_t.x + 3*x_a.x + 5*x_b.x + 4*x_r.x
                print(f"\nResource usage:")
                print(f"  Wood: {wood_used:.0f} / 890 units ({wood_used/890*100:.1f}%)")
                print(f"  Steel: {steel_used:.0f} / 500 units ({steel_used/500*100:.1f}%)")
                
                return {
                    'trucks': x_t.x, 'airplanes': x_a.x, 'boats': x_b.x, 'trains': x_r.x,
                    'profit': model.objVal, 'wood_used': wood_used, 'steel_used': steel_used
                }
                
            elif model.status == GRB.INFEASIBLE:
                print("Model is infeasible. Check constraints.")
                model.computeIIS()
                model.write("haus_toys_iis.ilp")
                print("IIS written to file 'haus_toys_iis.ilp'")
                
            elif model.status == GRB.UNBOUNDED:
                print("Model is unbounded.")
                
            elif model.status == GRB.INF_OR_UNBD:
                print("Model is infeasible or unbounded.")
                
            else:
                print(f"Optimization terminated with status {model.status}")
                
            return None
            
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

if __name__ == "__main__":
    solution = solve_haus_toys_problem()
    if solution:
        print(f"\nSolution verified. Total profit: ${solution['profit']:.2f}")