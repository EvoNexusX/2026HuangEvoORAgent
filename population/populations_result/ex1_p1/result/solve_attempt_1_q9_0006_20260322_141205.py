from gurobipy import Model, GRB

def solve_toy_production():
    """
    Solve the toy production optimization problem using Gurobi.
    
    This function models and solves the problem of determining the optimal
    production quantities of toy trucks, airplanes, boats, and trains
    to maximize profit, subject to resource constraints and logical conditions.
    """
    # Create a new optimization model
    m = Model("ToyProduction")
    
    # Compute tight upper bounds for each product from individual resource constraints
    # These bounds help determine a suitable big-M value for logical constraints
    max_T = min(890 // 12, 500 // 6)      # Max trucks considering wood and steel
    max_A = min(890 // 20, 500 // 3)      # Max airplanes
    max_B = min(890 // 15, 500 // 5)      # Max boats
    max_R = min(890 // 10, 500 // 4)      # Max trains
    
    # Set big-M value as the maximum of all upper bounds plus a safety margin
    M = max(max_T, max_A, max_B, max_R) + 1
    
    # Define decision variables
    T = m.addVar(lb=0, vtype=GRB.INTEGER, name="T")        # Number of trucks
    A = m.addVar(lb=0, vtype=GRB.INTEGER, name="A")        # Number of airplanes
    B = m.addVar(lb=0, vtype=GRB.INTEGER, name="B")        # Number of boats
    R = m.addVar(lb=0, vtype=GRB.INTEGER, name="R")        # Number of trains
    delta = m.addVar(vtype=GRB.BINARY, name="delta")       # Binary for truck-train logic
    y = m.addVar(vtype=GRB.BINARY, name="y")              # Binary for boat-airplane logic
    
    # Set objective function: maximize total profit
    profit = 5*T + 10*A + 8*B + 7*R
    m.setObjective(profit, GRB.MAXIMIZE)
    
    # Add resource constraints
    m.addConstr(12*T + 20*A + 15*B + 10*R <= 890, "WoodConstraint")
    m.addConstr(6*T + 3*A + 5*B + 4*R <= 500, "SteelConstraint")
    
    # Logical constraints
    # 1. If trucks are manufactured (T > 0), then trains cannot be manufactured (R = 0) and vice versa
    m.addConstr(T <= M * delta, "Logic1_Trucks")
    m.addConstr(R <= M * (1 - delta), "Logic1_Trains")
    
    # 2. If boats are manufactured (B > 0), then at least one airplane must be manufactured (A >= 1)
    m.addConstr(B <= M * y, "Logic2_Boats")
    m.addConstr(A >= y, "Logic2_Airplanes")  # When y = 1, forces A >= 1
    
    # 3. Number of boats cannot exceed number of trains
    m.addConstr(B <= R, "BoatsLE_Trains")
    
    # Optimize the model
    m.optimize()
    
    # Output results
    if m.status == GRB.OPTIMAL:
        print("Optimal solution found:")
        print(f"Total profit: ${m.objVal:.2f}")
        print(f"Toy trucks: {int(T.X)}")
        print(f"Toy airplanes: {int(A.X)}")
        print(f"Toy boats: {int(B.X)}")
        print(f"Toy trains: {int(R.X)}")
        print(f"Delta (truck-train indicator): {int(delta.X)}")
        print(f"Y (boat-airplane indicator): {int(y.X)}")
    elif m.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
    elif m.status == GRB.UNBOUNDED:
        print("Model is unbounded.")
    else:
        print(f"Optimization ended with status: {m.status}")

if __name__ == "__main__":
    solve_toy_production()