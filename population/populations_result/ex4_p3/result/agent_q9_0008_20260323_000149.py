# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

def solve_production_model():
    # Create model
    model = gp.Model("Production_Planning")
    
    # Decision variables
    x1 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x1")  # trucks
    x2 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x2")  # airplanes
    x3 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x3")  # boats
    x4 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x4")  # trains
    
    y1 = model.addVar(vtype=GRB.BINARY, name="y1")  # produce trucks?
    y2 = model.addVar(vtype=GRB.BINARY, name="y2")  # produce airplanes?
    y3 = model.addVar(vtype=GRB.BINARY, name="y3")  # produce boats?
    y4 = model.addVar(vtype=GRB.BINARY, name="y4")  # produce trains?
    
    # Set objective function
    model.setObjective(5*x1 + 10*x2 + 8*x3 + 7*x4, GRB.MAXIMIZE)
    
    # Resource constraints
    model.addConstr(12*x1 + 20*x2 + 15*x3 + 10*x4 <= 890, "wood")
    model.addConstr(6*x1 + 3*x2 + 5*x3 + 4*x4 <= 500, "steel")
    
    # Revised linking constraints with larger bounds
    bigM = 1000
    model.addConstr(x1 >= y1, "link_x1_lower")
    model.addConstr(x1 <= bigM * y1, "link_x1_upper")
    model.addConstr(x2 >= y2, "link_x2_lower")
    model.addConstr(x2 <= bigM * y2, "link_x2_upper")
    model.addConstr(x3 >= y3, "link_x3_lower")
    model.addConstr(x3 <= bigM * y3, "link_x3_upper")
    model.addConstr(x4 >= y4, "link_x4_lower")
    model.addConstr(x4 <= bigM * y4, "link_x4_upper")
    
    # Logical constraints - corrected modeling
    model.addConstr(y1 + y4 <= 1, "truck_train_exclusive")
    model.addConstr(y3 <= y2, "boat_requires_airplane")
    model.addConstr(x3 <= x4, "boat_leq_train")
    
    # Additional logical consistency constraint
    # If x3 > 0 then y3 = 1, handled by x3 >= y3
    # If x4 > 0 then y4 = 1, handled by x4 >= y4
    
    # Solve settings
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2  # Allow non-convex constraints
    
    # Optimize
    model.optimize()
    
    # Results
    if model.status == GRB.OPTIMAL:
        print(f"Maximum profit: ${model.objVal:.2f}")
        print("Optimal production plan:")
        print(f"  Trucks: {int(x1.x)} (produce: {int(y1.x)})")
        print(f"  Airplanes: {int(x2.x)} (produce: {int(y2.x)})")
        print(f"  Boats: {int(x3.x)} (produce: {int(y3.x)})")
        print(f"  Trains: {int(x4.x)} (produce: {int(y4.x)})")
        print("\nResource usage:")
        wood_used = 12*x1.x + 20*x2.x + 15*x3.x + 10*x4.x
        steel_used = 6*x1.x + 3*x2.x + 5*x3.x + 4*x4.x
        print(f"  Wood: {wood_used:.0f} / 890 units")
        print(f"  Steel: {steel_used:.0f} / 500 units")
    elif model.status == GRB.INFEASIBLE:
        print("Model is infeasible. Relaxing constraints...")
        
        # Compute and print the IIS to diagnose infeasibility
        model.computeIIS()
        model.write("model_iis.ilp")
        print("Infeasible constraints written to model_iis.ilp")
        
    else:
        print(f"No optimal solution found. Status: {model.status}")
    
    return model

if __name__ == "__main__":
    solve_production_model()