import gurobipy as gp

# Create model
model = gp.Model("Haus_Toys_Production")

# Parameters
profit = {"Truck": 5, "Airplane": 10, "Boat": 8, "Train": 7}
wood_use = {"Truck": 12, "Airplane": 20, "Boat": 15, "Train": 10}
steel_use = {"Truck": 6, "Airplane": 3, "Boat": 5, "Train": 4}
wood_total = 890
steel_total = 500

# Calculate upper bounds
U_T = min(wood_total // wood_use["Truck"], steel_total // steel_use["Truck"])
U_A = min(wood_total // wood_use["Airplane"], steel_total // steel_use["Airplane"])
U_B = min(wood_total // wood_use["Boat"], steel_total // steel_use["Boat"])
U_R = min(wood_total // wood_use["Train"], steel_total // steel_use["Train"])

# Decision variables
x_T = model.addVar(lb=0, ub=U_T, vtype=gp.GRB.INTEGER, name="x_T")
x_A = model.addVar(lb=0, ub=U_A, vtype=gp.GRB.INTEGER, name="x_A")
x_B = model.addVar(lb=0, ub=U_B, vtype=gp.GRB.INTEGER, name="x_B")
x_R = model.addVar(lb=0, ub=U_R, vtype=gp.GRB.INTEGER, name="x_R")
y_T = model.addVar(vtype=gp.GRB.BINARY, name="y_T")
y_B = model.addVar(vtype=gp.GRB.BINARY, name="y_B")

# Objective function
model.setObjective(
    profit["Truck"] * x_T + 
    profit["Airplane"] * x_A + 
    profit["Boat"] * x_B + 
    profit["Train"] * x_R, 
    gp.GRB.MAXIMIZE
)

# Constraints
model.addConstr(
    wood_use["Truck"] * x_T + 
    wood_use["Airplane"] * x_A + 
    wood_use["Boat"] * x_B + 
    wood_use["Train"] * x_R <= wood_total, 
    "Wood_Constraint"
)

model.addConstr(
    steel_use["Truck"] * x_T + 
    steel_use["Airplane"] * x_A + 
    steel_use["Boat"] * x_B + 
    steel_use["Train"] * x_R <= steel_total, 
    "Steel_Constraint"
)

model.addConstr(x_T <= U_T * y_T, "Truck_Indicator_Upper")
model.addConstr(x_T >= y_T, "Truck_Indicator_Lower")
model.addConstr(x_B <= U_B * y_B, "Boat_Indicator_Upper")
model.addConstr(x_B >= y_B, "Boat_Indicator_Lower")
model.addConstr(x_R <= U_R * (1 - y_T), "Truck_Train_Exclusive")
model.addConstr(x_A >= y_B, "Boat_Airplane_Link")
model.addConstr(x_B <= x_R, "Boat_Train_Quantity")

# Solve
model.optimize()

# Results
if model.status == gp.GRB.OPTIMAL:
    print(f'Optimal profit: ${model.objVal:.2f}')
    print(f'Trucks produced: {x_T.x:.0f}')
    print(f'Airplanes produced: {x_A.x:.0f}')
    print(f'Boats produced: {x_B.x:.0f}')
    print(f'Trains produced: {x_R.x:.0f}')
    print(f'Produce trucks (y_T): {y_T.x:.0f}')
    print(f'Produce boats (y_B): {y_B.x:.0f}')
else:
    print('No optimal solution found')