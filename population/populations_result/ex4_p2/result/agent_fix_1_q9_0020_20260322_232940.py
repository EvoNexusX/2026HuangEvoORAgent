import gurobipy as gp
from gurobipy import GRB

def solve_haus_toys():
    """
    Solves the Haus Toys production optimization problem using Gurobi.
    Returns the optimal solution if found, otherwise handles infeasible/unbounded cases.
    """
    try:
        # Create model
        model = gp.Model("HausToysProduction")
        
        # Create variables
        # Production quantities (integer)
        x_t = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_t")  # trucks
        x_a = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_a")  # airplanes
        x_b = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_b")  # boats
        x_r = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_r")  # trains
        
        # Binary indicator variables
        y_t = model.addVar(vtype=GRB.BINARY, name="y_t")
        y_a = model.addVar(vtype=GRB.BINARY, name="y_a")
        y_b = model.addVar(vtype=GRB.BINARY, name="y_b")
        y_r = model.addVar(vtype=GRB.BINARY, name="y_r")
        
        # Set objective: maximize profit
        profit = 5*x_t + 10*x_a + 8*x_b + 7*x_r
        model.setObjective(profit, GRB.MAXIMIZE)
        
        # Add constraints
        
        # 1. Resource constraints
        model.addConstr(12*x_t + 20*x_a + 15*x_b + 10*x_r <= 890, "wood_availability")
        model.addConstr(6*x_t + 3*x_a + 5*x_b + 4*x_r <= 500, "steel_availability")
        
        # 2. Logical constraints
        model.addConstr(y_t + y_r <= 1, "truck_train_exclusive")
        model.addConstr(y_b <= y_a, "boat_implies_airplane")
        model.addConstr(x_b <= x_r, "boats_not_exceed_trains")
        
        # 3. Linking constraints (connect production variables with binary indicators)
        # Upper bounds calculated from resource constraints
        model.addConstr(x_t <= 74 * y_t, "link_truck_upper")
        model.addConstr(x_t >= y_t, "link_truck_lower")
        
        model.addConstr(x_a <= 44 * y_a, "link_airplane_upper")
        model.addConstr(x_a >= y_a, "link_airplane_lower")
        
        model.addConstr(x_b <= 59 * y_b, "link_boat_upper")
        model.addConstr(x_b >= y_b, "link_boat_lower")
        
        model.addConstr(x_r <= 89 * y_r, "link_train_upper")
        model.addConstr(x_r >= y_r, "link_train_lower")
        
        # Solve the model
        model.optimize()
        
        # Check and display solution status
        if model.status == GRB.OPTIMAL:
            print(f"Optimal solution found with maximum profit: ${model.objVal:.2f}\n")
            
            # Display production plan
            print("Production Plan:")
            print(f"  Toy Trucks: {x_t.x:.0f} units")
            print(f"  Toy Airplanes: {x_a.x:.0f} units")
            print(f"  Toy Boats: {x_b.x:.0f} units")
            print(f"  Toy Trains: {x_r.x:.0f} units")
            
            # Display resource usage
            wood_used = 12*x_t.x + 20*x_a.x + 15*x_b.x + 10*x_r.x
            steel_used = 6*x_t.x + 3*x_a.x + 5*x_b.x + 4*x_r.x
            print(f"\nResource Usage:")
            print(f"  Wood: {wood_used:.0f} / 890 units ({wood_used/890*100:.1f}%)")
            print(f"  Steel: {steel_used:.0f} / 500 units ({steel_used/500*100:.1f}%)")
            
            # Display binary variable values for logical constraints verification
            print(f"\nIndicator Variables:")
            print(f"  y_t (trucks): {y_t.x:.0f}")
            print(f"  y_a (airplanes): {y_a.x:.0f}")
            print(f"  y_b (boats): {y_b.x:.0f}")
            print(f"  y_r (trains): {y_r.x:.0f}")
            
            return {
                'trucks': x_t.x,
                'airplanes': x_a.x,
                'boats': x_b.x,
                'trains': x_r.x,
                'profit': model.objVal,
                'wood_used': wood_used,
                'steel_used': steel_used
            }
            
        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible. No solution satisfies all constraints.")
            # Compute IIS to help debug infeasibility
            model.computeIIS()
            model.write("haus_toys_iis.ilp")
            print("Irreducible Inconsistent Subsystem (IIS) written to 'haus_toys_iis.ilp'")
            
        elif model.status == GRB.UNBOUNDED:
            print("Model is unbounded. Objective can be increased indefinitely.")
            
        elif model.status == GRB.INF_OR_UNBD:
            print("Model is infeasible or unbounded. Try setting DualReductions=0.")
            
        elif model.status == GRB.TIME_LIMIT:
            print("Time limit reached. Best solution found:")
            if model.solCount > 0:
                print(f"  Best profit found: ${model.objVal:.2f}")
            else:
                print("  No feasible solution found within time limit.")
                
        else:
            print(f"Optimization ended with status: {model.status}")
            
        return None
        
    except gp.GurobiError as e:
        print(f"Gurobi error encountered: {e}")
        return None
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

if __name__ == "__main__":
    solve_haus_toys()