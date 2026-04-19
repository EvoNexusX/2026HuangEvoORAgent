import gurobipy as gp
from gurobipy import GRB

# Define data
warehouses = ['Verona', 'Perugia', 'Rome', 'Pescara', 'Taranto', 'Lamezia']
ports = ['Genoa', 'Venice', 'Ancona', 'Naples', 'Bari']

# Supply and demand arrays
supply = [10, 12, 20, 24, 18, 40]
demand = [20, 15, 25, 33, 21]

# Distance matrix (in km) from warehouses to ports
# Order: [Genoa, Venice, Ancona, Naples, Bari]
distances = [
    [290, 115, 355, 715, 810],   # Verona
    [380, 340, 165, 380, 610],   # Perugia
    [505, 530, 285, 220, 450],   # Rome
    [655, 450, 155, 240, 315],   # Pescara
    [1010, 840, 550, 305, 95],   # Taranto
    [1072, 1097, 747, 372, 333]  # Lamezia
]

# Create model
model = gp.Model("Container_Transportation")

# Create decision variables: x[i,j] = containers from warehouse i to port j
x = {}
for i in range(len(warehouses)):
    for j in range(len(ports)):
        # Continuous variables - will yield integer solution due to problem structure
        x[i, j] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, 
                              name=f"x_{warehouses[i]}_{ports[j]}")

# Set objective: minimize total transportation cost
# Cost per container = 30 * distance
model.setObjective(
    gp.quicksum(30 * distances[i][j] * x[i, j] 
                for i in range(len(warehouses)) 
                for j in range(len(ports))),
    GRB.MINIMIZE
)

# Add supply constraints: containers shipped from each warehouse ≤ available supply
for i in range(len(warehouses)):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(len(ports))) <= supply[i],
        name=f"Supply_{warehouses[i]}"
    )

# Add demand constraints: containers received at each port = demand
for j in range(len(ports)):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(len(warehouses))) == demand[j],
        name=f"Demand_{ports[j]}"
    )

# Optimize the model
model.optimize()

# Check and display results
if model.status == GRB.OPTIMAL:
    print(f"Optimal solution found!")
    print(f"Total transportation cost: €{model.objVal:,.2f}")
    print("\nTransportation plan (non-zero flows):")
    
    # Track totals for verification
    total_shipped = 0
    print(f"{'From':<10} {'To':<10} {'Containers':>12} {'Distance':>10} {'Cost':>15}")
    print("-" * 65)
    
    for i in range(len(warehouses)):
        for j in range(len(ports)):
            if x[i, j].X > 0.001:  # Display non-zero flows
                flow = x[i, j].X
                distance = distances[i][j]
                cost = 30 * distance * flow
                total_shipped += flow
                print(f"{warehouses[i]:<10} {ports[j]:<10} {flow:12.0f} {distance:10}km €{cost:14,.2f}")
    
    print("-" * 65)
    print(f"\nSummary:")
    print(f"Total containers shipped: {total_shipped:.0f}")
    print(f"Total supply available: {sum(supply)}")
    print(f"Total demand: {sum(demand)}")
    
    # Verify constraints
    print("\nConstraint verification:")
    for i in range(len(warehouses)):
        shipped = sum(x[i, j].X for j in range(len(ports)))
        print(f"{warehouses[i]}: shipped {shipped:.0f} of {supply[i]} available")
    
    for j in range(len(ports)):
        received = sum(x[i, j].X for i in range(len(warehouses)))
        print(f"{ports[j]}: received {received:.0f} (demand: {demand[j]})")

elif model.status == GRB.INFEASIBLE:
    print("Model is infeasible!")
elif model.status == GRB.UNBOUNDED:
    print("Model is unbounded!")
else:
    print(f"Optimization terminated with status: {model.status}")