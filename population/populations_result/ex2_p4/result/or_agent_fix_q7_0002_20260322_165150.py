import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("ContainerTransport")

# Data definition
warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]

supply = [10, 12, 20, 24, 18, 40]
demand = [20, 15, 25, 33, 21]

# Distance matrix (km)
distances = [
    [290, 115, 355, 715, 810],
    [380, 340, 165, 380, 610],
    [505, 530, 285, 220, 450],
    [655, 450, 155, 240, 315],
    [1010, 840, 550, 305, 95],
    [1072, 1097, 747, 372, 333]
]

# Cost matrix (EUR/container)
cost_per_km = 30
costs = [[dist * cost_per_km for dist in row] for row in distances]

# Create variables
x = model.addVars(len(warehouses), len(ports), lb=0, vtype=GRB.CONTINUOUS, name="x")

# Set objective function
model.setObjective(
    gp.quicksum(
        costs[i][j] * x[i, j]
        for i in range(len(warehouses))
        for j in range(len(ports))
    ),
    GRB.MINIMIZE
)

# Add supply constraints
for i in range(len(warehouses)):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(len(ports))) <= supply[i],
        name=f"Supply_{warehouses[i]}"
    )

# Add demand constraints
for j in range(len(ports)):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(len(warehouses))) == demand[j],
        name=f"Demand_{ports[j]}"
    )

# Solve model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Optimal total cost: {model.objVal:.2f} EUR")
    print("\nOptimal transportation plan:")
    for i in range(len(warehouses)):
        for j in range(len(ports)):
            if x[i, j].x > 1e-6:
                print(f"{warehouses[i]} -> {ports[j]}: {x[i, j].x:.0f} containers")
    
    # Calculate and output truck count
    total_containers = sum(demand)
    trucks_needed = (total_containers + 1) // 2  # ceiling division
    print(f"\nTrucks needed: {trucks_needed} trucks (each truck carries up to 2 containers)")
else:
    print("No optimal solution found")