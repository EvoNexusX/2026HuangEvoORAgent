import gurobipy as gp
from gurobipy import GRB

# Define the problem data
warehouses = ['Verona', 'Perugia', 'Rome', 'Pescara', 'Taranto', 'Lamezia']
ports = ['Genoa', 'Venice', 'Ancona', 'Naples', 'Bari']

# Supply at each warehouse
supply = [10, 12, 20, 24, 18, 40]

# Demand at each port
demand = [20, 15, 25, 33, 21]

# Distance matrix (km) - warehouses × ports
distances = [
    [290, 115, 355, 715, 810],    # Verona to ports
    [380, 340, 165, 380, 610],    # Perugia to ports
    [505, 530, 285, 220, 450],    # Rome to ports
    [655, 450, 155, 240, 315],    # Pescara to ports
    [1010, 840, 550, 305, 95],    # Taranto to ports
    [1072, 1097, 747, 372, 333]   # Lamezia to ports
]

# Create the optimization model
model = gp.Model("Container_Transportation")

# Create decision variables: x[i,j] = containers from warehouse i to port j
x = {}
for i in range(len(warehouses)):
    for j in range(len(ports)):
        var_name = f"x_{warehouses[i]}_{ports[j]}"
        x[i, j] = model.addVar(lb=0.0, name=var_name, vtype=GRB.CONTINUOUS)

# Set objective: minimize total transportation cost
# Cost per container = 30 euros/km × distance
objective = gp.quicksum(
    30 * distances[i][j] * x[i, j] 
    for i in range(len(warehouses)) 
    for j in range(len(ports))
)
model.setObjective(objective, GRB.MINIMIZE)

# Add supply constraints: total from each warehouse ≤ supply
for i in range(len(warehouses)):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(len(ports))) <= supply[i],
        name=f"Supply_{warehouses[i]}"
    )

# Add demand constraints: total to each port = demand
for j in range(len(ports)):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(len(warehouses))) == demand[j],
        name=f"Demand_{ports[j]}"
    )

# Optimize the model
model.optimize()

# Analyze and display the solution
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Total transportation cost: €{model.ObjVal:.2f}")
    print("\nOptimal transportation plan (containers):")
    print("=" * 60)
    
    # Display the shipping plan in a table format
    header = "From/To\t\t" + "\t".join(ports)
    print(header)
    print("-" * 60)
    
    for i in range(len(warehouses)):
        row = f"{warehouses[i]:<10}"
        for j in range(len(ports)):
            if x[i, j].x > 0.001:  # Show non-zero values
                row += f"\t{x[i, j].x:.0f}"
            else:
                row += "\t0"
        print(row)
    
    print("\n" + "=" * 60)
    print("\nDetailed shipment information:")
    print("-" * 60)
    
    total_cost = 0
    for i in range(len(warehouses)):
        for j in range(len(ports)):
            if x[i, j].x > 0.001:
                cost = 30 * distances[i][j] * x[i, j].x
                total_cost += cost
                print(f"{warehouses[i]} -> {ports[j]}: {x[i, j].x:.0f} containers, "
                      f"distance: {distances[i][j]} km, cost: €{cost:.2f}")
    
    print("\n" + "=" * 60)
    print("\nWarehouse utilization:")
    print("-" * 60)
    for i in range(len(warehouses)):
        shipped = sum(x[i, j].x for j in range(len(ports)))
        utilization = (shipped / supply[i]) * 100 if supply[i] > 0 else 0
        print(f"{warehouses[i]}: {shipped:.0f}/{supply[i]} containers "
              f"({utilization:.1f}% utilized)")
    
    print("\nPort demand fulfillment:")
    print("-" * 60)
    for j in range(len(ports)):
        received = sum(x[i, j].x for i in range(len(warehouses)))
        print(f"{ports[j]}: {received:.0f}/{demand[j]} containers "
              f"({(received/demand[j])*100:.1f}% fulfilled)")
    
    print("\n" + "=" * 60)
    print(f"Total cost verification: €{total_cost:.2f}")
    
elif model.status == GRB.INFEASIBLE:
    print("Model is infeasible!")
    # Compute and display the IIS (Irreducible Inconsistent Subsystem)
    model.computeIIS()
    print("\nIIS constraints:")
    for c in model.getConstrs():
        if c.IISConstr:
            print(f"  {c.ConstrName}")
    
elif model.status == GRB.UNBOUNDED:
    print("Model is unbounded!")
    
elif model.status == GRB.INF_OR_UNBD:
    print("Model is infeasible or unbounded!")
    
else:
    print(f"Optimization ended with status: {model.status}")