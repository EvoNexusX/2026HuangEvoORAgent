import gurobipy as gp
from gurobipy import GRB

# Data definition
warehouses = ['Verona', 'Perugia', 'Rome', 'Pescara', 'Taranto', 'Lamezia']
ports = ['Genoa', 'Venice', 'Ancona', 'Naples', 'Bari']

supply = [10, 12, 20, 24, 18, 40]
demand = [20, 15, 25, 33, 21]

# Distance matrix (km)
dist_matrix = [
    [290, 115, 355, 715, 810],    # Verona
    [380, 340, 165, 380, 610],    # Perugia
    [505, 530, 285, 220, 450],    # Rome
    [655, 450, 155, 240, 315],    # Pescara
    [1010, 840, 550, 305, 95],    # Taranto
    [1072, 1097, 747, 372, 333]   # Lamezia
]

# Create model
model = gp.Model("ContainerTransport")

# Create decision variables
x = {}
for i in range(len(warehouses)):
    for j in range(len(ports)):
        x[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{warehouses[i]}_{ports[j]}")

# Set objective function: minimize total cost
model.setObjective(
    gp.quicksum(30 * dist_matrix[i][j] * x[i, j] 
                for i in range(len(warehouses)) 
                for j in range(len(ports))),
    GRB.MINIMIZE
)

# Add supply constraints
for i in range(len(warehouses)):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(len(ports))) <= supply[i],
        name=f"supply_{warehouses[i]}"
    )

# Add demand constraints
for j in range(len(ports)):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(len(warehouses))) == demand[j],
        name=f"demand_{ports[j]}"
    )

# Solve the model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Optimal total cost: {model.ObjVal:.2f} euros\n")
    print("Transportation plan:")
    total_containers = 0
    for i in range(len(warehouses)):
        for j in range(len(ports)):
            if x[i, j].X > 0:
                print(f"  From {warehouses[i]:8} to {ports[j]:8}: {x[i, j].X:2} containers")
                total_containers += x[i, j].X
    print(f"\nTotal containers transported: {total_containers}")
else:
    print("No optimal solution found")
    print(f"Solver status: {model.status}")