import gurobipy as gp
from gurobipy import GRB

def solve_transportation_problem():
    try:
        # Data
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
        
        # Distance matrix in km
        distances = {
            "Verona": {"Genoa": 290, "Venice": 115, "Ancona": 355, "Naples": 715, "Bari": 810},
            "Perugia": {"Genoa": 380, "Venice": 340, "Ancona": 165, "Naples": 380, "Bari": 610},
            "Rome": {"Genoa": 505, "Venice": 530, "Ancona": 285, "Naples": 220, "Bari": 450},
            "Pescara": {"Genoa": 655, "Venice": 450, "Ancona": 155, "Naples": 240, "Bari": 315},
            "Taranto": {"Genoa": 1010, "Venice": 840, "Ancona": 550, "Naples": 305, "Bari": 95},
            "Lamezia": {"Genoa": 1072, "Venice": 1097, "Ancona": 747, "Naples": 372, "Bari": 333}
        }
        
        # Cost per km
        cost_per_km = 30
        
        # Create model
        model = gp.Model("ContainerTransportation")
        
        # Create decision variables
        x = {}
        for i in warehouses:
            for j in ports:
                var_name = f"x_{i}_{j}"
                x[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=var_name)
        
        # Set objective function
        model.setObjective(
            gp.quicksum(
                cost_per_km * distances[i][j] * x[i, j] 
                for i in warehouses 
                for j in ports
            ),
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
        
        # Optimize model
        model.optimize()
        
        # Check and display solution status
        if model.status == GRB.OPTIMAL:
            print(f"Optimal solution found!")
            print(f"Total transportation cost: €{model.objVal:,.2f}")
            print("\nOptimal transportation plan:")
            print("-" * 50)
            
            total_containers = 0
            for i in warehouses:
                for j in ports:
                    if x[i, j].x > 0:
                        cost = cost_per_km * distances[i][j] * x[i, j].x
                        print(f"{i:10} -> {j:10}: {int(x[i, j].x):2d} containers "
                              f"(distance: {distances[i][j]:4d} km, cost: €{cost:,.2f})")
                        total_containers += x[i, j].x
            
            print("-" * 50)
            print(f"Total containers transported: {total_containers}")
            
            # Check supply and demand satisfaction
            print("\nSupply utilization:")
            for i in warehouses:
                used = sum(x[i, j].x for j in ports)
                print(f"{i:10}: {used}/{supply[i]} containers used")
            
            print("\nDemand satisfaction:")
            for j in ports:
                received = sum(x[i, j].x for i in warehouses)
                print(f"{j:10}: {received}/{demand[j]} containers received")
                
        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible!")
            model.computeIIS()
            model.write("model_iis.ilp")
            print("Irreducible Inconsistent Subsystem written to model_iis.ilp")
            
        elif model.status == GRB.UNBOUNDED:
            print("Model is unbounded!")
            
        elif model.status == GRB.INF_OR_UNBD:
            print("Model is infeasible or unbounded!")
            
        else:
            print(f"Optimization ended with status {model.status}")
            
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    solve_transportation_problem()