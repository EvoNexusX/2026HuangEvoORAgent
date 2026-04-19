import gurobipy as gp
from gurobipy import GRB

# Create a new model
model = gp.Model("HausToys")

# Decision variables
T = model.addVar(vtype=GRB.INTEGER, lb=0, name="T")
A = model.addVar(vtype=GRB.INTEGER, lb=0, name="A")
B = model.addVar(vtype=GRB.INTEGER, lb=0, name="B")
R = model.addVar(vtype=GRB.INTEGER, lb=0, name="R")
y_T = model.addVar(vtype=GRB.BINARY, name="y_T")
y_A = model.addVar(vtype=GRB.BINARY, name="y_A")
y_B = model.addVar(vtype=GRB.BINARY, name="y_B")
y_R = model.addVar(vtype=GRB.BINARY, name="y_R")

# Set objective
model.setObjective(5*T + 10*A + 8*B + 7*R, GRB.MAXIMIZE)

# Resource constraints
model.addConstr(12*T + 20*A + 15*B + 10*R <= 890, "Wood")
model.addConstr(6*T + 3*A + 5*B + 4*R <= 500, "Steel")

# Linking constraints
model.addConstr(T >= y_T, "Link_T_lower")
model.addConstr(T <= 74 * y_T, "Link_T_upper")
model.addConstr(A >= y_A, "Link_A_lower")
model.addConstr(A <= 44 * y_A, "Link_A_upper")
model.addConstr(B >= y_B, "Link_B_lower")
model.addConstr(B <= 59 * y_B, "Link_B_upper")
model.addConstr(R >= y_R, "Link_R_lower")
model.addConstr(R <= 89 * y_R, "Link_R_upper")

# Logical constraints
model.addConstr(y_T + y_R <= 1, "Truck_Train_exclusive")
model.addConstr(y_B <= y_A, "Boat_implies_Airplane")
model.addConstr(B <= R, "Boat_leq_Train")

# Optimize model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Optimal solution found with total profit: ${model.ObjVal:.2f}")
    print(f"Number of trucks (T): {T.X}")
    print(f"Number of airplanes (A): {A.X}")
    print(f"Number of boats (B): {B.X}")
    print(f"Number of trains (R): {R.X}")
    print(f"Indicator y_T: {y_T.X}")
    print(f"Indicator y_A: {y_A.X}")
    print(f"Indicator y_B: {y_B.X}")
    print(f"Indicator y_R: {y_R.X}")
else:
    print(f"Optimization failed with status: {model.status}")