# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

# Data preparation
warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]

supply = [10, 12, 20, 24, 18, 40]
demand = [20, 15, 25, 33, 21]

distances = [
    [290, 115, 355, 715, 810],
    [380, 340, 165, 380, 610],
    [505, 530, 285, 220, 450],
    [655, 450, 155, 240, 315],
    [1010, 840, 550, 305, 95],
    [1072, 1097, 747, 372, 333]
]

cost_per_km = 30
truck_capacity = 2

# Create model
model = gp.Model("ContainerTransportation")

# Define decision variables
x = model.addVars(len(warehouses), len(ports), vtype=GRB.INTEGER, name="x")
t = model.addVars(len(warehouses), len(ports), vtype=GRB.INTEGER, name="t")

# Set objective function
model.setObjective(
    gp.quicksum(
        cost_per_km * distances[i][j] * t[i, j]
        for i in range(len(warehouses))
        for j in range(len(ports))
    ),
    GRB.MINIMIZE
)

# Add constraints
# 1. Supply constraints
for i in range(len(warehouses)):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(len(ports))) <= supply[i],
        name=f"Supply_{warehouses[i]}"
    )

# 2. Demand constraints
for j in range(len(ports)):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(len(warehouses))) == demand[j],
        name=f"Demand_{ports[j]}"
    )

# 3. Truck capacity constraints
for i in range(len(warehouses)):
    for j in range(len(ports)):
        model.addConstr(
            x[i, j] <= truck_capacity * t[i, j],
            name=f"Capacity_{warehouses[i]}_{ports[j]}"
        )

# Solve parameters
model.Params.OutputFlag = 1
model.Params.TimeLimit = 300
model.Params.MIPGap = 0.01

# Solve model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Optimal total cost: {model.objVal:.2f} euros")
    print("\nTransportation plan details:")
    for i in range(len(warehouses)):
        for j in range(len(ports)):
            if x[i, j].x > 0:
                transport_cost = cost_per_km * distances[i][j] * t[i, j].x
                print(f"{warehouses[i]} -> {ports[j]}: {int(x[i, j].x)} containers, {int(t[i, j].x)} trucks, cost: {transport_cost:.2f} euros")
    
    print("\nSupply utilization:")
    for i in range(len(warehouses)):
        used = sum(x[i, j].x for j in range(len(ports)))
        print(f"{warehouses[i]}: {int(used)}/{supply[i]} containers")
    
    print("\nDemand satisfaction:")
    for j in range(len(ports)):
        received = sum(x[i, j].x for i in range(len(warehouses)))
        print(f"{ports[j]}: {int(received)}/{demand[j]} containers")
    
    total_trucks = sum(t[i, j].x for i in range(len(warehouses)) for j in range(len(ports)))
    print(f"\nTotal trucks used: {int(total_trucks)}")
else:
    print("No optimal solution found")
    print(f"Solution status: {model.status}")