import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("ContainerTransportation")

# Warehouse set
I = ['Verona', 'Perugia', 'Rome', 'Pescara', 'Taranto', 'Lamezia']

# Port set
J = ['Genoa', 'Venice', 'Ancona', 'Naples', 'Bari']

# Supply parameters
s = {'Verona': 10, 'Perugia': 12, 'Rome': 20, 'Pescara': 24, 'Taranto': 18, 'Lamezia': 40}

# Demand parameters
d = {'Genoa': 20, 'Venice': 15, 'Ancona': 25, 'Naples': 33, 'Bari': 21}

# Distance matrix
dist = {
    ('Verona', 'Genoa'): 290,
    ('Verona', 'Venice'): 115,
    ('Verona', 'Ancona'): 355,
    ('Verona', 'Naples'): 715,
    ('Verona', 'Bari'): 810,
    ('Perugia', 'Genoa'): 380,
    ('Perugia', 'Venice'): 340,
    ('Perugia', 'Ancona'): 165,
    ('Perugia', 'Naples'): 380,
    ('Perugia', 'Bari'): 610,
    ('Rome', 'Genoa'): 505,
    ('Rome', 'Venice'): 530,
    ('Rome', 'Ancona'): 285,
    ('Rome', 'Naples'): 220,
    ('Rome', 'Bari'): 450,
    ('Pescara', 'Genoa'): 655,
    ('Pescara', 'Venice'): 450,
    ('Pescara', 'Ancona'): 155,
    ('Pescara', 'Naples'): 240,
    ('Pescara', 'Bari'): 315,
    ('Taranto', 'Genoa'): 1010,
    ('Taranto', 'Venice'): 840,
    ('Taranto', 'Ancona'): 550,
    ('Taranto', 'Naples'): 305,
    ('Taranto', 'Bari'): 95,
    ('Lamezia', 'Genoa'): 1072,
    ('Lamezia', 'Venice'): 1097,
    ('Lamezia', 'Ancona'): 747,
    ('Lamezia', 'Naples'): 372,
    ('Lamezia', 'Bari'): 333,
}

# Cost parameters
c = 30
CAP = 2

# Create decision variables
x = model.addVars(I, J, vtype=GRB.INTEGER, lb=0, name="x")
y = model.addVars(I, J, vtype=GRB.INTEGER, lb=0, name="y")

# Supply constraints
supply_constr = model.addConstrs((x.sum(i, '*') <= s[i] for i in I), name="Supply")

# Demand constraints
demand_constr = model.addConstrs((x.sum('*', j) == d[j] for j in J), name="Demand")

# Truck capacity constraints
capacity_constr = model.addConstrs((x[i, j] <= CAP * y[i, j] for i in I for j in J), name="Capacity")

# Objective function
objective = gp.quicksum(c * dist[i, j] * y[i, j] for i in I for j in J)
model.setObjective(objective, GRB.MINIMIZE)

# Solve model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Optimal solution found.")
    print(f"Total cost: {model.objVal:.2f} euros")
    print("\nContainer flow (x_ij):")
    for i in I:
        for j in J:
            val = x[i, j].X
            if val > 0:
                print(f"  {i} -> {j}: {int(val)} containers")
    print("\nTruck usage (y_ij):")
    total_cost = 0
    for i in I:
        for j in J:
            val = y[i, j].X
            if val > 0:
                route_cost = c * dist[i, j] * val
                total_cost += route_cost
                print(f"  {i} -> {j}: {int(val)} trucks (distance {dist[i, j]} km, cost {route_cost:.2f} euros)")
    print(f"\nTotal cost verification: {total_cost:.2f} euros")
else:
    print(f"No optimal solution found. Status: {model.status}")