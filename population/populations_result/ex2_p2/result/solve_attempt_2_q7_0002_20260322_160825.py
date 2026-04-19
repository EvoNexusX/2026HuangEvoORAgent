# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

def solve_container_transport():
    """
    Solves the container transportation optimization problem using Gurobi.
    """
    try:
        # 1. Create model
        model = gp.Model("ContainerTransport")
        
        # 2. Data preparation
        # Warehouses (indices 0-5)
        warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
        I = list(range(len(warehouses)))
        
        # Ports (indices 0-4)
        ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]
        J = list(range(len(ports)))
        
        # Supply (empty containers)
        supply = [10, 12, 20, 24, 18, 40]
        
        # Demand (container requirements)
        demand = [20, 15, 25, 33, 21]
        
        # Distance matrix (km) - rows: warehouses, columns: ports
        distances = [
            [290, 115, 355, 715, 810],   # Verona
            [380, 340, 165, 380, 610],   # Perugia
            [505, 530, 285, 220, 450],   # Rome
            [655, 450, 155, 240, 315],   # Pescara
            [1010, 840, 550, 305, 95],   # Taranto
            [1072, 1097, 747, 372, 333]  # Lamezia
        ]
        
        # Transport cost parameter (EUR/km)
        cost_per_km = 30
        
        # 3. Decision variables
        # x_ij: number of trucks from warehouse i to port j (integer)
        x = model.addVars(I, J, vtype=GRB.INTEGER, name="trucks")
        # y_ij: number of containers from warehouse i to port j (integer)
        y = model.addVars(I, J, vtype=GRB.INTEGER, name="containers")
        
        # 4. Objective function
        # Minimize total transport cost = sum(cost_per_km * distance_ij * trucks_ij)
        model.setObjective(
            gp.quicksum(cost_per_km * distances[i][j] * x[i, j] 
                       for i in I for j in J),
            GRB.MINIMIZE
        )
        
        # 5. Constraints
        # Supply constraint: total containers shipped from each warehouse <= supply
        model.addConstrs(
            (y.sum(i, '*') <= supply[i] for i in I),
            name="supply"
        )
        
        # Demand constraint: total containers received at each port = demand
        model.addConstrs(
            (y.sum('*', j) == demand[j] for j in J),
            name="demand"
        )
        
        # Truck capacity constraint: each truck can carry at most 2 containers
        model.addConstrs(
            (y[i, j] <= 2 * x[i, j] for i in I for j in J),
            name="capacity"
        )
        
        # 6. Solve model
        model.optimize()
        
        # 7. Results output
        print("=" * 60)
        print("Container Transportation Optimization - Results")
        print("=" * 60)
        
        if model.status == GRB.OPTIMAL:
            print(f"\nOptimal total cost: {model.objVal:,.2f} EUR")
            
            # Summary by warehouse
            print("\n" + "=" * 60)
            print("Transportation summary by warehouse:")
            print("=" * 60)
            for i in I:
                total_containers = sum(y[i, j].x for j in J)
                total_trucks = sum(x[i, j].x for j in J)
                if total_containers > 0:
                    print(f"\n{warehouses[i]} (supply: {supply[i]}):")
                    print(f"  Containers shipped: {total_containers}")
                    print(f"  Trucks used: {total_trucks}")
                    for j in J:
                        if y[i, j].x > 0:
                            cost = cost_per_km * distances[i][j] * x[i, j].x
                            print(f"  -> {ports[j]}: {int(y[i, j].x)} containers, "
                                  f"{int(x[i, j].x)} trucks, cost: {cost:,.2f} EUR")
            
            # Summary by port
            print("\n" + "=" * 60)
            print("Receiving summary by port:")
            print("=" * 60)
            for j in J:
                total_received = sum(y[i, j].x for i in I)
                print(f"\n{ports[j]} (demand: {demand[j]}):")
                print(f"  Containers received: {total_received}")
                for i in I:
                    if y[i, j].x > 0:
                        print(f"  <- {warehouses[i]}: {int(y[i, j].x)} containers")
            
            # Detailed transportation plan table
            print("\n" + "=" * 60)
            print("Detailed transportation plan:")
            print("=" * 60)
            print("\n{:<10} {:<10} {:<10} {:<10} {:<15}".format(
                "Warehouse", "Port", "Containers", "Trucks", "Cost(EUR)"
            ))
            print("-" * 60)
            total_containers_shipped = 0
            total_trucks_used = 0
            
            for i in I:
                for j in J:
                    if y[i, j].x > 0:
                        containers = int(y[i, j].x)
                        trucks = int(x[i, j].x)
                        cost = cost_per_km * distances[i][j] * trucks
                        total_containers_shipped += containers
                        total_trucks_used += trucks
                        
                        print("{:<10} {:<10} {:<10} {:<10} {:<15,.2f}".format(
                            warehouses[i], ports[j], containers, trucks, cost
                        ))
            
            print("-" * 60)
            print(f"Total: {total_containers_shipped} containers, "
                  f"{total_trucks_used} trucks")
            
        elif model.status == GRB.INFEASIBLE:
            print("Problem is infeasible! Please check constraints.")
        elif model.status == GRB.UNBOUNDED:
            print("Problem is unbounded!")
        else:
            print(f"Solution status: {model.status}")
            
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
    except Exception as e:
        print(f"Program error: {e}")

if __name__ == "__main__":
    solve_container_transport()