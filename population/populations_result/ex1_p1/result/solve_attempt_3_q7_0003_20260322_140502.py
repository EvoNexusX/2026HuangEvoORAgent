import gurobipy as gp
from gurobipy import GRB

# ---------- Data Definition ----------
# Warehouse and port names
warehouses = ['Verona', 'Perugia', 'Rome', 'Pescara', 'Taranto', 'Lamezia']
ports = ['Genoa', 'Venice', 'Ancona', 'Naples', 'Bari']

# Supply and demand (in same order as above)
supply = [10, 12, 20, 24, 18, 40]
demand = [20, 15, 25, 33, 21]

# Distance matrix (6 warehouses x 5 ports) from the problem description
# Format: distances[warehouse_index][port_index]
distances = [
    [290, 115, 355, 715, 810],   # Verona
    [380, 340, 165, 380, 610],   # Perugia
    [505, 530, 285, 220, 450],   # Rome
    [655, 450, 155, 240, 315],   # Pescara
    [1010, 840, 550, 305, 95],   # Taranto
    [1072, 1097, 747, 372, 333]  # Lamezia
]

# Compute cost matrix: cost per container = 30 euros/km * distance
cost = [[30 * distances[i][j] for j in range(len(ports))] 
        for i in range(len(warehouses))]

# ---------- Model Construction ----------
model = gp.Model('ContainerTransportation')

# Decision variables: x[i, j] = containers from warehouse i to port j
x = {}
for i in range(len(warehouses)):
    for j in range(len(ports)):
        x[i, j] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, 
                               name=f'x_{warehouses[i]}_{ports[j]}')

# Objective: minimize total transportation cost
obj = gp.quicksum(cost[i][j] * x[i, j] 
                  for i in range(len(warehouses)) 
                  for j in range(len(ports)))
model.setObjective(obj, GRB.MINIMIZE)

# Supply constraints: cannot exceed available containers at each warehouse
for i in range(len(warehouses)):
    model.addConstr(gp.quicksum(x[i, j] for j in range(len(ports))) <= supply[i],
                    name=f'Supply_{warehouses[i]}')

# Demand constraints: must exactly meet demand at each port
for j in range(len(ports)):
    model.addConstr(gp.quicksum(x[i, j] for i in range(len(warehouses))) == demand[j],
                    name=f'Demand_{ports[j]}')

# ---------- Solve and Output ----------
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f'Optimal total cost: €{model.objVal:,.2f}')
    print('\nOptimal shipment plan (containers):')
    
    # Print non-zero flows in a formatted table
    print(f"{'From':<10} {'To':<10} {'Containers':<12} {'Cost per Container':<20} {'Total Cost':<15}")
    print("-" * 70)
    
    total_cost = 0
    for i in range(len(warehouses)):
        for j in range(len(ports)):
            flow = x[i, j].x
            if flow > 1e-6:  # Only print non-zero flows
                cost_per = cost[i][j]
                flow_cost = flow * cost_per
                total_cost += flow_cost
                print(f"{warehouses[i]:<10} {ports[j]:<10} {flow:<12.0f} "
                      f"€{cost_per:<19,.0f} €{flow_cost:<14,.0f}")
    
    print("-" * 70)
    print(f"{'Total cost:':<32} €{total_cost:,.0f}")
    
    # Check supply utilization
    print("\nSupply utilization:")
    for i in range(len(warehouses)):
        used = sum(x[i, j].x for j in range(len(ports)))
        print(f"{warehouses[i]:<10} Used: {used:.0f}/{supply[i]} "
              f"({used/supply[i]*100:.1f}%)")
    
    # Check demand satisfaction
    print("\nDemand satisfaction:")
    for j in range(len(ports)):
        received = sum(x[i, j].x for i in range(len(warehouses)))
        print(f"{ports[j]:<10} Received: {received:.0f}/{demand[j]}")
        
elif model.status == GRB.INFEASIBLE:
    print("Model is infeasible.")
    # Compute IIS to identify conflicting constraints
    model.computeIIS()
    model.write("model_iis.ilp")
    print("Infeasible constraints written to 'model_iis.ilp'")
else:
    print(f"Optimization ended with status: {model.status}")

# Optional: write model to file for inspection
# model.write('transportation.lp')