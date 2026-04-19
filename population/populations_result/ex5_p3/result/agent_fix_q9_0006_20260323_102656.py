# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("Haus_Toys_Production")

# Define decision variables
x1 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x1")  # trucks
x2 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x2")  # airplanes
x3 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x3")  # boats
x4 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x4")  # trains

y1 = model.addVar(vtype=GRB.BINARY, name="y1")  # produce trucks?
y2 = model.addVar(vtype=GRB.BINARY, name="y2")  # produce airplanes?
y3 = model.addVar(vtype=GRB.BINARY, name="y3")  # produce boats?
y4 = model.addVar(vtype=GRB.BINARY, name="y4")  # produce trains?

# Update model to add variables
model.update()

# Set objective function: maximize profit
model.setObjective(5*x1 + 10*x2 + 8*x3 + 7*x4, GRB.MAXIMIZE)

# Add resource constraints
# Wood constraint: 12*x1 + 20*x2 + 15*x3 + 10*x4 <= 890
model.addConstr(12*x1 + 20*x2 + 15*x3 + 10*x4 <= 890, "Wood_Constraint")

# Steel constraint: 6*x1 + 3*x2 + 5*x3 + 4*x4 <= 500
model.addConstr(6*x1 + 3*x2 + 5*x3 + 4*x4 <= 500, "Steel_Constraint")

# Big-M constraints: link production to binary variables
# Big-M values: M1=74, M2=44, M3=59, M4=89
M1 = 74
M2 = 44
M3 = 59
M4 = 89

model.addConstr(x1 <= M1 * y1, "BigM_x1_upper")
model.addConstr(x1 >= y1, "BigM_x1_lower")
model.addConstr(x2 <= M2 * y2, "BigM_x2_upper")
model.addConstr(x2 >= y2, "BigM_x2_lower")
model.addConstr(x3 <= M3 * y3, "BigM_x3_upper")
model.addConstr(x3 >= y3, "BigM_x3_lower")
model.addConstr(x4 <= M4 * y4, "BigM_x4_upper")
model.addConstr(x4 >= y4, "BigM_x4_lower")

# Add logical constraints
# Trucks and trains are mutually exclusive: y1 + y4 <= 1
model.addConstr(y1 + y4 <= 1, "Truck_Train_Exclusive")

# If boats are produced, airplanes must be produced: y3 <= y2
model.addConstr(y3 <= y2, "Boat_requires_Airplane")

# Number of boats cannot exceed number of trains: x3 <= x4
model.addConstr(x3 <= x4, "Boat_leq_Train_Quantity")

# Solve model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Maximum profit: ${model.objVal:.2f}")
    print("\nOptimal production plan:")
    print(f"  Trucks (x1): {x1.x} units")
    print(f"  Airplanes (x2): {x2.x} units")
    print(f"  Boats (x3): {x3.x} units")
    print(f"  Trains (x4): {x4.x} units")
    print(f"  Produce trucks? (y1): {int(y1.x)}")
    print(f"  Produce airplanes? (y2): {int(y2.x)}")
    print(f"  Produce boats? (y3): {int(y3.x)}")
    print(f"  Produce trains? (y4): {int(y4.x)}")
    
    # Verify resource usage
    wood_used = 12*x1.x + 20*x2.x + 15*x3.x + 10*x4.x
    steel_used = 6*x1.x + 3*x2.x + 5*x3.x + 4*x4.x
    print(f"\nResource usage:")
    print(f"  Wood used: {wood_used} / 890 units")
    print(f"  Steel used: {steel_used} / 500 units")
else:
    print("Optimal solution not found. Status:", model.status)