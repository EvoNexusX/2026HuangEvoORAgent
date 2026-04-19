import gurobipy as gp
from gurobipy import GRB

def main():
    """
    Solve the Haus Toys production optimization problem using Gurobi.
    """
    try:
        # 1. Create model
        model = gp.Model("HausToys")
        
        # 2. Define decision variables
        # Continuous variables for production quantities
        x1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x1_truck")
        x2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x2_airplane")
        x3 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x3_boat")
        x4 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x4_train")
        
        # Binary variables for production decisions
        y1 = model.addVar(vtype=GRB.BINARY, name="y1_truck")
        y2 = model.addVar(vtype=GRB.BINARY, name="y2_airplane")
        y3 = model.addVar(vtype=GRB.BINARY, name="y3_boat")
        y4 = model.addVar(vtype=GRB.BINARY, name="y4_train")
        
        # 3. Set objective function: maximize total profit
        model.setObjective(5*x1 + 10*x2 + 8*x3 + 7*x4, GRB.MAXIMIZE)
        
        # 4. Add constraints
        # Resource constraints
        model.addConstr(12*x1 + 20*x2 + 15*x3 + 10*x4 <= 890, "wood_constraint")
        model.addConstr(6*x1 + 3*x2 + 5*x3 + 4*x4 <= 500, "steel_constraint")
        
        # Production indicator linking constraints
        model.addConstr(x1 <= 75*y1, "link_truck")
        model.addConstr(x2 <= 45*y2, "link_airplane")
        model.addConstr(x3 <= 60*y3, "link_boat")
        model.addConstr(x4 <= 90*y4, "link_train")
        
        # Logical constraints
        model.addConstr(y1 + y4 <= 1, "truck_train_exclusion")
        model.addConstr(y3 <= y2, "boat_implies_airplane")
        
        # Quantity relationship constraint
        model.addConstr(x3 <= x4, "boat_le_train")
        
        # 5. Solve the model
        model.optimize()
        
        # 6. Output results
        if model.status == GRB.OPTIMAL:
            print("\n=== OPTIMAL SOLUTION FOUND ===\n")
            print(f"Maximum Profit: ${model.objVal:.2f}\n")
            print("Production Quantities:")
            print(f"  Toy Trucks (x1): {x1.x:.2f}")
            print(f"  Toy Airplanes (x2): {x2.x:.2f}")
            print(f"  Toy Boats (x3): {x3.x:.2f}")
            print(f"  Toy Trains (x4): {x4.x:.2f}\n")
            print("Production Decisions (1=produce, 0=not produce):")
            print(f"  Produce Trucks (y1): {int(y1.x)}")
            print(f"  Produce Airplanes (y2): {int(y2.x)}")
            print(f"  Produce Boats (y3): {int(y3.x)}")
            print(f"  Produce Trains (y4): {int(y4.x)}")
        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        elif model.status == GRB.UNBOUNDED:
            print("Model is unbounded.")
        else:
            print(f"Optimization terminated with status {model.status}")
            
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()