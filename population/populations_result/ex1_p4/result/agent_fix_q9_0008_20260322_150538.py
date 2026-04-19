import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("HausToys")

# Decision variables
x1 = model.addVar(vtype=GRB.INTEGER, name="x1")  # trucks
x2 = model.addVar(vtype=GRB.INTEGER, name="x2")  # airplanes
x3 = model.addVar(vtype=GRB.INTEGER, name="x3")  # boats
x4 = model.addVar(vtype=GRB.INTEGER, name="x4")  # trains
y = model.addVar(vtype=GRB.BINARY, name="y")     # truck-train indicator
z = model.addVar(vtype=GRB.BINARY, name="z")     # boat indicator

# Set objective function
model.setObjective(5*x1 + 10*x2 + 8*x3 + 7*x4, GRB.MAXIMIZE)

# Add constraints
# Resource constraints
model.addConstr(12*x1 + 20*x2 + 15*x3 + 10*x4 <= 890, "Wood")
model.addConstr(6*x1 + 3*x2 + 5*x3 + 4*x4 <= 500, "Steel")

# Truck-train exclusivity constraint
M1 = 74
M2 = 89
model.addConstr(x1 <= M1 * y, "Truck_activation")
model.addConstr(x4 <= M2 * (1 - y), "Train_activation")

# Boat-airplane logic constraint
M3 = 59
model.addConstr(x3 <= M3 * z, "Boat_activation")
model.addConstr(x2 >= z, "Airplane_if_boat")

# Boat-train quantity constraint
model.addConstr(x3 <= x4, "Boat_vs_Train")

# Solve model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    print(f"Number of trucks (x1): {x1.X}")
    print(f"Number of airplanes (x2): {x2.X}")
    print(f"Number of boats (x3): {x3.X}")
    print(f"Number of trains (x4): {x4.X}")
    print(f"Indicator y (1 if trucks are made): {y.X}")
    print(f"Indicator z (1 if boats are made): {z.X}")
    print(f"Maximum profit: ${model.ObjVal}")
else:
    print("No optimal solution found. Status:", model.status)