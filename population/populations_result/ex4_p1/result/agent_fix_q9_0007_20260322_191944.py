import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("Toy_Production")

# Define decision variables
x1 = model.addVar(lb=0, vtype=GRB.INTEGER, name="x1")  # trucks
x2 = model.addVar(lb=0, vtype=GRB.INTEGER, name="x2")  # airplanes
x3 = model.addVar(lb=0, vtype=GRB.INTEGER, name="x3")  # boats
x4 = model.addVar(lb=0, vtype=GRB.INTEGER, name="x4")  # trains

b1 = model.addVar(vtype=GRB.BINARY, name="b1")  # produce trucks
b2 = model.addVar(vtype=GRB.BINARY, name="b2")  # produce airplanes
b3 = model.addVar(vtype=GRB.BINARY, name="b3")  # produce boats
b4 = model.addVar(vtype=GRB.BINARY, name="b4")  # produce trains

# Set objective function
model.setObjective(5*x1 + 10*x2 + 8*x3 + 7*x4, GRB.MAXIMIZE)

# Add constraints

# Resource constraints
model.addConstr(12*x1 + 20*x2 + 15*x3 + 10*x4 <= 890, "Wood")
model.addConstr(6*x1 + 3*x2 + 5*x3 + 4*x4 <= 500, "Steel")

# Big-M constraints for linking variables
M1 = 74
M2 = 44
M3 = 59
M4 = 89

model.addConstr(x1 <= M1 * b1, "M1_upper")
model.addConstr(x1 >= b1, "M1_lower")

model.addConstr(x2 <= M2 * b2, "M2_upper")
model.addConstr(x2 >= b2, "M2_lower")

model.addConstr(x3 <= M3 * b3, "M3_upper")
model.addConstr(x3 >= b3, "M3_lower")

model.addConstr(x4 <= M4 * b4, "M4_upper")
model.addConstr(x4 >= b4, "M4_lower")

# Logical constraints
model.addConstr(b4 <= 1 - b1, "No_train_if_truck")
model.addConstr(b2 >= b3, "If_boat_then_airplane")
model.addConstr(x3 <= x4, "Boat_no_more_than_train")

# Solve the model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Optimal objective value: {model.objVal}")
    print(f"Number of trucks: {x1.x}")
    print(f"Number of airplanes: {x2.x}")
    print(f"Number of boats: {x3.x}")
    print(f"Number of trains: {x4.x}")
    print(f"Produce trucks (b1): {b1.x}")
    print(f"Produce airplanes (b2): {b2.x}")
    print(f"Produce boats (b3): {b3.x}")
    print(f"Produce trains (b4): {b4.x}")
else:
    print("No optimal solution found")