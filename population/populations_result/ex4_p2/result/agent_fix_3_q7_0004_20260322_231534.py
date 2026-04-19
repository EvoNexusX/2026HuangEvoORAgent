import gurobipy as gp
from gurobipy import GRB

def solve_container_transport():
    try:
        # Data definition
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
        
        # Distance matrix in km (warehouse -> port)
        distance = {
            "Verona": {"Genoa": 290, "Venice": 115, "Ancona": 355, "Naples": 715, "Bari": 810},
            "Perugia": {"Genoa": 380, "Venice": 340, "Ancona": 165, "Naples": 380, "Bari": 610},
            "Rome": {"Genoa": 505, "Venice": 530, "Ancona": 285, "Naples": 220, "Bari": 450},
            "Pescara": {"Genoa": 655, "Venice": 450, "Ancona": 155, "Naples": 240, "Bari": 315},
            "Taranto": {"Genoa": 1010, "Venice": 840, "Ancona": 550, "Naples": 305, "Bari": 95},
            "Lamezia": {"Genoa": 1072, "Venice": 1097, "Ancona": 747, "Naples": 372, "Bari": 333}
        }
        
        # Calculate cost matrix (30 euros per km per container)
        cost_per_km = 30
        cost = {}
        for i in warehouses:
            cost[i] = {}
            for j in ports:
                cost[i][j] = cost_per_km * distance[i][j]
        
        # Create model
        model = gp.Model("ContainerTransport")
        
        # Create decision variables
        x = {}
        for i in warehouses:
            for j in ports:
                x[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{i}_{j}")
        
        # Set objective function: minimize total transportation cost
        model.setObjective(
            gp.quicksum(cost[i][j] * x[i, j] for i in warehouses for j in ports),
            GRB.MINIMIZE
        )
        
        # Add supply constraints
        for i in warehouses:
            model.addConstr(
                gp.quicksum(x[i, j] for j in ports) <= supply[i],
                name=f"supply_{i}"
            )
        
        # Add demand constraints
        for j in ports:
            model.addConstr(
                gp.quicksum(x[i, j] for i in warehouses) == demand[j],
                name=f"demand_{j}"
            )
        
        # Set optimization parameters
        model.Params.OutputFlag = 1
        model.Params.MIPGap = 1e-6
        
        # Solve the model
        model.optimize()
        
        # Check optimization status and display results
        if model.status == GRB.OPTIMAL:
            print(f"Optimal solution found with total cost: {model.objVal:.2f} euros\n")
            print("Transportation plan (non-zero flows):")
            print("-" * 50)
            total_containers = 0
            for i in warehouses:
                for j in ports:
                    if x[i, j].x > 0.5:  # Non-zero flow (accounting for floating point)
                        print(f"{i:10s} -> {j:10s}: {int(x[i, j].x):2d} containers")
                        total_containers += int(x[i, j].x)
            print("-" * 50)
            print(f"Total containers transported: {total_containers}")
            
            # Verify constraints
            print("\nConstraint verification:")
            print("Supply utilization:")
            for i in warehouses:
                used = sum(int(x[i, j].x) for j in ports)
                print(f"  {i:10s}: {used:2d}/{supply[i]}")
            
            print("\nDemand satisfaction:")
            for j in ports:
                received = sum(int(x[i, j].x) for i in warehouses)
                print(f"  {j:10s}: {received:2d}/{demand[j]}")
                
        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
            # Compute and print IIS if needed
            model.computeIIS()
            print("Irreducible Inconsistent Subsystem (IIS):")
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"  Constraint: {c.constrName}")
        elif model.status == GRB.UNBOUNDED:
            print("Model is unbounded.")
        elif model.status == GRB.INF_OR_UNBD:
            print("Model is infeasible or unbounded.")
        else:
            print(f"Optimization ended with status: {model.status}")
            
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    solve_container_transport()