# -*- coding: utf-8 -*-
from gurobipy import *

# Define warehouse and port indices
warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]
I = range(len(warehouses))
J = range(len(ports))

# Supply and demand data
supply = [10, 12, 20, 24, 18, 40]
demand = [20, 15, 25, 33, 21]

# Distance matrix (km)
dist = [
    [290, 115, 355, 715, 810],
    [380, 340, 165, 380, 610],
    [505, 530, 285, 220, 450],
    [655, 450, 155, 240, 315],
    [1010, 840, 550, 305, 95],
    [1072, 1097, 747, 372, 333]
]

# Calculate cost matrix: 30 euros per km per container
cost = [[30 * dist[i][j] for j in J] for i in I]

# Create model
model = Model("ContainerTransportation")

# Decision variables: number of containers from warehouse i to port j
x = model.addVars(I, J, lb=0.0, name="x")

# Set objective: minimize total transportation cost
model.setObjective(quicksum(cost[i][j] * x[i, j] for i in I for j in J), GRB.MINIMIZE)

# Supply constraints: from each warehouse
model.addConstrs((x.sum(i, '*') <= supply[i] for i in I), name="Supply")

# Demand constraints: to each port
model.addConstrs((x.sum('*', j) == demand[j] for j in J), name="Demand")

# Solve the model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Minimum total cost: {model.objVal:.2f} euros")
    print("\nOptimal transportation plan (non-zero flows):")
    total_containers = 0
    for i in I:
        for j in J:
            if x[i, j].x > 1e-6:
                print(f"{warehouses[i]:<8} -> {ports[j]:<8}: {x[i, j].x:3.0f} containers")
                total_containers += x[i, j].x
    print(f"\nTotal containers transported: {total_containers:.0f}")
else:
    print("No optimal solution found")