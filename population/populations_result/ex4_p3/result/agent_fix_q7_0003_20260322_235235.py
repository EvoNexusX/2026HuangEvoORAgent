import gurobipy as gp
from gurobipy import GRB

# Warehouse and port information
warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]

# Supply and demand
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

# Transport cost per km (euros)
cost_per_km = 30

# Truck capacity
truck_capacity = 2

# Create model
model = gp.Model("EmptyContainerTransportation")

# Create variables
x = {}  # Number of containers from warehouse i to port j
y = {}  # Number of trucks from warehouse i to port j

for i in range(len(warehouses)):
    for j in range(len(ports)):
        x[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{i}_{j}")
        y[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"y_{i}_{j}")

# Set objective function: minimize total transport cost
model.setObjective(
    gp.quicksum(
        cost_per_km * distances[i][j] * y[i, j]
        for i in range(len(warehouses))
        for j in range(len(ports))
    ),
    GRB.MINIMIZE
)

# Add constraints

# Supply constraints: total containers from each warehouse <= supply
for i in range(len(warehouses)):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(len(ports))) <= supply[i],
        name=f"Supply_{i}"
    )

# Demand constraints: total containers to each port = demand
for j in range(len(ports)):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(len(warehouses))) == demand[j],
        name=f"Demand_{j}"
    )

# Truck capacity constraints: each truck can carry at most 2 containers
for i in range(len(warehouses)):
    for j in range(len(ports)):
        model.addConstr(
            x[i, j] <= truck_capacity * y[i, j],
            name=f"Truck_{i}_{j}"
        )

# Solve model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Minimum transport cost: {model.ObjVal:.2f} euros")
    print("\nContainer transport plan:")
    print("Warehouse -> Port : Containers (Trucks)")
    
    for i in range(len(warehouses)):
        for j in range(len(ports)):
            if x[i, j].X > 0:
                print(f"{warehouses[i]} -> {ports[j]} : {int(x[i, j].X)} ({int(y[i, j].X)})")
    
    print("\nWarehouse supply usage:")
    for i in range(len(warehouses)):
        total_sent = sum(x[i, j].X for j in range(len(ports)))
        print(f"{warehouses[i]}: Sent {int(total_sent)} / Supply {supply[i]}")
    
    print("\nPort demand satisfaction:")
    for j in range(len(ports)):
        total_received = sum(x[i, j].X for i in range(len(warehouses)))
        print(f"{ports[j]}: Received {int(total_received)} / Demand {demand[j]}")
else:
    print("Optimal solution not found")
    print(f"Model status: {model.status}")