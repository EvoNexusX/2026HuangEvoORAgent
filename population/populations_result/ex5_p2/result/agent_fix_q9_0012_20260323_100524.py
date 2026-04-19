# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

# Create environment and model
env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()
model = gp.Model("HausToys_Production", env=env)

# Define variables
x_T = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_T")
x_A = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_A")
x_B = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_B")
x_R = model.addVar(vtype=GRB.INTEGER, lb=0, name="x_R")

y_T = model.addVar(vtype=GRB.BINARY, name="y_T")
y_A = model.addVar(vtype=GRB.BINARY, name="y_A")
y_B = model.addVar(vtype=GRB.BINARY, name="y_B")
y_R = model.addVar(vtype=GRB.BINARY, name="y_R")

# Set objective function
model.setObjective(5*x_T + 10*x_A + 8*x_B + 7*x_R, GRB.MAXIMIZE)

# Resource constraints
model.addConstr(12*x_T + 20*x_A + 15*x_B + 10*x_R <= 890, "wood")
model.addConstr(6*x_T + 3*x_A + 5*x_B + 4*x_R <= 500, "steel")

# Logic constraints
model.addConstr(y_T + y_R <= 1, "truck_train_excl")
model.addConstr(y_B <= y_A, "boat_implies_airplane")
model.addConstr(x_B <= x_R, "boat_leq_train")

# Big-M constraints with refined bounds
M_T = 74  # min(890/12, 500/6) rounded down
M_A = 44  # min(890/20, 500/3) rounded down
M_B = 59  # min(890/15, 500/5) rounded down
M_R = 89  # min(890/10, 500/4) rounded down

model.addConstr(x_T <= M_T * y_T, "bigM_T_upper")
model.addConstr(x_A <= M_A * y_A, "bigM_A_upper")
model.addConstr(x_B <= M_B * y_B, "bigM_B_upper")
model.addConstr(x_R <= M_R * y_R, "bigM_R_upper")

model.addConstr(x_T >= y_T, "bigM_T_lower")
model.addConstr(x_A >= y_A, "bigM_A_lower")
model.addConstr(x_B >= y_B, "bigM_B_lower")
model.addConstr(x_R >= y_R, "bigM_R_lower")

# Solve the model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Optimal Profit: ${model.objVal:.2f}")
    print(f"Trucks: {x_T.x:.0f}")
    print(f"Airplanes: {x_A.x:.0f}")
    print(f"Boats: {x_B.x:.0f}")
    print(f"Trains: {x_R.x:.0f}")
    print(f"\nProduction Indicators:")
    print(f"  Produce Trucks (y_T): {y_T.x:.0f}")
    print(f"  Produce Airplanes (y_A): {y_A.x:.0f}")
    print(f"  Produce Boats (y_B): {y_B.x:.0f}")
    print(f"  Produce Trains (y_R): {y_R.x:.0f}")
elif model.status == GRB.INFEASIBLE:
    print("Model is infeasible")
    model.computeIIS()
    model.write("model.ilp")
    print("IIS written to model.ilp for diagnosis")
else:
    print(f"Optimization ended with status: {model.status}")

# Clean up
model.dispose()
env.dispose()