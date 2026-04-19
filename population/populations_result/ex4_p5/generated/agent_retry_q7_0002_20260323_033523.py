import gurobipy as gp
from gurobipy import GRB

# Step 2: Initialize model
model = gp.Model("Empty_Container_Transportation")

# Step 3: Define sets and parameters
warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]

supply = {
    "Verona": 10,
    "Perugia": 12,
    "Rome": 20,
    "Pescara": 24,
    "Taranto": 18,
    "Lamezia": 40
}

demand = {
    "Genoa": 20,
    "Venice": 15,
    "Ancona": 25,
    "Naples": 33,
    "Bari": 21
}

distances = {
    ("Verona", "Genoa"): 290,
    ("Verona", "Venice"): 115,
    ("Verona", "Ancona"): 355,
    ("Verona", "Naples"): 715,
    ("Verona", "Bari"): 810,
    ("Perugia", "Genoa"): 380,
    ("Perugia", "Venice"): 340,
    ("Perugia", "Ancona"): 165,
    ("Perugia", "Naples"): 380,
    ("Perugia", "Bari"): 610,
    ("Rome", "Genoa"): 505,
    ("Rome", "Venice"): 530,
    ("Rome", "Ancona"): 285,
    ("Rome", "Naples"): 220,
    ("Rome", "Bari"): 450,
    ("Pescara", "Genoa"): 655,
    ("Pescara", "Venice"): 450,
    ("Pescara", "Ancona"): 155,
    ("Pescara", "Naples"): 240,
    ("Pescara", "Bari"): 315,
    ("Taranto", "Genoa"): 1010,
    ("Taranto", "Venice"): 840,
    ("Taranto", "Ancona"): 550,
    ("Taranto", "Naples"): 305,
    ("Taranto", "Bari"): 95,
    ("Lamezia", "Genoa"): 1072,
    ("Lamezia", "Venice"): 1097,
    ("Lamezia", "Ancona"): 747,
    ("Lamezia", "Naples"): 372,
    ("Lamezia", "Bari"): 333,
}

c = 30  # Cost rate per km (euros/km)

# Step 4: Add decision variables
x = {}
y = {}

for i in warehouses:
    for j in ports:
        x[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{i}_{j}")
        y[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"y_{i}_{j}")

# Step 5: Set objective function
model.setObjective(
    gp.quicksum(c * distances[i, j] * y[i, j] for i in warehouses for j in ports),
    GRB.MINIMIZE
)

# Step 6: Add constraints
# Supply constraints
for i in warehouses:
    model.addConstr(
        gp.quicksum(x[i, j] for j in ports) <= supply[i],
        name=f"Supply_{i}"
    )

# Demand constraints
for j in ports:
    model.addConstr(
        gp.quicksum(x[i, j] for i in warehouses) == demand[j],
        name=f"Demand_{j}"
    )

# Truck capacity and minimum load constraints
for i in warehouses:
    for j in ports:
        model.addConstr(x[i, j] <= 2 * y[i, j], name=f"Capacity_{i}_{j}")
        model.addConstr(x[i, j] >= y[i, j], name=f"MinLoad_{i}_{j}")

# Step 7: Configure solver parameters (optional)
model.setParam('OutputFlag', 1)  # Show solving process

# Step 8: Solve the model
model.optimize()

# Step 9: Extract and analyze results
if model.status == GRB.OPTIMAL:
    print(f"\nOptimal total transportation cost: {model.ObjVal:.2f} euros")
    
    # Output non-zero transportation flows
    print("\nNon-zero transportation plan:")
    total_containers = 0
    for i in warehouses:
        for j in ports:
            if x[i, j].X > 0:
                containers = x[i, j].X
                trucks = y[i, j].X
                cost_per_route = c * distances[i, j] * trucks
                total_containers += containers
                print(f"  From {i} to {j}: {containers} containers, {trucks} trucks, route cost: {cost_per_route:.2f} euros")
    
    print(f"\nTotal transported containers: {total_containers}")
    print(f"Total demand: {sum(demand.values())}")
    print(f"Total supply: {sum(supply.values())}")
    
    # Verify demand satisfaction
    for j in ports:
        received = sum(x[i, j].X for i in warehouses)
        print(f"  Port {j}: demand {demand[j]}, received {received}, {'satisfied' if received == demand[j] else 'not satisfied'}")
    
    # Check truck constraints
    print("\nTruck usage verification:")
    for i in warehouses:
        for j in ports:
            if x[i, j].X > 0:
                containers = x[i, j].X
                trucks = y[i, j].X
                if not (containers >= trucks and containers <= 2 * trucks):
                    print(f"  Warning: Route {i}-{j} violates truck constraints")
                else:
                    print(f"  Route {i}-{j}: {containers} containers, {trucks} trucks (satisfies {trucks} <= {containers} <= {2*trucks})")
else:
    print("Optimal solution not found")
    print(f"Solution status: {model.status}")