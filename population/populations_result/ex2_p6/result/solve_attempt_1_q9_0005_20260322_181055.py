import gurobipy as gp
from gurobipy import GRB

def solve_haus_toys():
    """
    Solves the Haus Toys manufacturing optimization problem using Gurobi.
    """
    try:
        # --- Step 1: Create Model ---
        model = gp.Model("HausToys")
        
        # --- Step 2: Define Variables ---
        # Integer variables for toy quantities
        x_T = model.addVar(vtype=GRB.INTEGER, lb=0, name="Trucks")
        x_A = model.addVar(vtype=GRB.INTEGER, lb=0, name="Airplanes")
        x_B = model.addVar(vtype=GRB.INTEGER, lb=0, name="Boats")
        x_Tn = model.addVar(vtype=GRB.INTEGER, lb=0, name="Trains")
        
        # Binary indicator variables
        y_T = model.addVar(vtype=GRB.BINARY, name="y_T")
        y_Tn = model.addVar(vtype=GRB.BINARY, name="y_Tn")
        y_B = model.addVar(vtype=GRB.BINARY, name="y_B")
        
        # --- Step 3: Set Objective ---
        # Maximize total profit
        model.setObjective(5*x_T + 10*x_A + 8*x_B + 7*x_Tn, GRB.MAXIMIZE)
        
        # --- Step 4: Add Constraints ---
        # 1. Wood constraint (units: 890 available)
        model.addConstr(12*x_T + 20*x_A + 15*x_B + 10*x_Tn <= 890, "Wood")
        
        # 2. Steel constraint (units: 500 available)
        model.addConstr(6*x_T + 3*x_A + 5*x_B + 4*x_Tn <= 500, "Steel")
        
        # 3-4. Truck indicator constraints (big-M method)
        # Upper bound: if y_T=0, x_T=0; if y_T=1, x_T <= 74 (from wood limit)
        model.addConstr(x_T <= 74 * y_T, "Truck_Upper")
        # Lower bound: if y_T=1, x_T >= 1
        model.addConstr(x_T >= y_T, "Truck_Lower")
        
        # 5-6. Train indicator constraints
        model.addConstr(x_Tn <= 89 * y_Tn, "Train_Upper")
        model.addConstr(x_Tn >= y_Tn, "Train_Lower")
        
        # 7-8. Boat indicator constraints
        model.addConstr(x_B <= 59 * y_B, "Boat_Upper")
        model.addConstr(x_B >= y_B, "Boat_Lower")
        
        # 9. Mutual exclusivity: trucks and trains cannot both be manufactured
        model.addConstr(y_T + y_Tn <= 1, "No_Truck_Train")
        
        # 10. Boats imply airplanes: if y_B=1, then x_A >= 1
        model.addConstr(x_A >= y_B, "Boats_Imply_Airplanes")
        
        # 11. Boats do not exceed trains
        model.addConstr(x_B <= x_Tn, "Boats_LE_Trains")
        
        # --- Step 5: Solve the Model ---
        model.optimize()
        
        # --- Step 6: Extract and Display Results ---
        if model.status == GRB.OPTIMAL:
            print(f"Optimal Profit: ${model.ObjVal:.2f}")
            print("\nProduction Plan:")
            print(f"  Trucks: {int(x_T.X)}")
            print(f"  Airplanes: {int(x_A.X)}")
            print(f"  Boats: {int(x_B.X)}")
            print(f"  Trains: {int(x_Tn.X)}")
            print(f"\nIndicator Variables:")
            print(f"  y_T (Trucks manufactured): {int(y_T.X)}")
            print(f"  y_Tn (Trains manufactured): {int(y_Tn.X)}")
            print(f"  y_B (Boats manufactured): {int(y_B.X)}")
        else:
            print(f"Model status: {model.status}")
            if model.status == GRB.INFEASIBLE:
                print("Model is infeasible. Consider relaxing constraints.")
                # Compute and display the Irreducible Inconsistent Subsystem (IIS) for debugging
                model.computeIIS()
                model.write("haus_toys_iis.ilp")
                print("IIS written to haus_toys_iis.ilp")
            elif model.status == GRB.UNBOUNDED:
                print("Model is unbounded. Check objective and constraints.")
            elif model.status == GRB.INF_OR_UNBD:
                print("Model is infeasible or unbounded.")
                
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    solve_haus_toys()