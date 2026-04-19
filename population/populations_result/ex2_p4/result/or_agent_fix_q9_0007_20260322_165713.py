import gurobipy as gp
from gurobipy import GRB

model = gp.Model("Toy_Manufacturing")

T = model.addVar(vtype=GRB.INTEGER, lb=0, name="T")
A = model.addVar(vtype=GRB.INTEGER, lb=0, name="A")
B = model.addVar(vtype=GRB.INTEGER, lb=0, name="B")
R = model.addVar(vtype=GRB.INTEGER, lb=0, name="R")

y_T = model.addVar(vtype=GRB.BINARY, name="y_T")
y_A = model.addVar(vtype=GRB.BINARY, name="y_A")
y_B = model.addVar(vtype=GRB.BINARY, name="y_B")
y_R = model.addVar(vtype=GRB.BINARY, name="y_R")

model.setObjective(5*T + 10*A + 8*B + 7*R, GRB.MAXIMIZE)

model.addConstr(12*T + 20*A + 15*B + 10*R <= 890, "Wood")
model.addConstr(6*T + 3*A + 5*B + 4*R <= 500, "Steel")
model.addConstr(y_T + y_R <= 1, "Truck_Train_Exclusive")
model.addConstr(y_B <= y_A, "Boat_Implies_Airplane")
model.addConstr(B <= R, "Boat_leq_Train")
model.addConstr(T <= 74 * y_T, "Ind_T")
model.addConstr(A <= 44 * y_A, "Ind_A")
model.addConstr(B <= 59 * y_B, "Ind_B")
model.addConstr(R <= 89 * y_R, "Ind_R")

model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal objective value: $%.2f" % model.ObjVal)
    print("Production quantities:")
    print("  Trucks: %d" % T.X)
    print("  Airplanes: %d" % A.X)
    print("  Boats: %d" % B.X)
    print("  Trains: %d" % R.X)
    print("Production indicators:")
    print("  Trucks: %d" % y_T.X)
    print("  Airplanes: %d" % y_A.X)
    print("  Boats: %d" % y_B.X)
    print("  Trains: %d" % y_R.X)
else:
    print("No optimal solution found")