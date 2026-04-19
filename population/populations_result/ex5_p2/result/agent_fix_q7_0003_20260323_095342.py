# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

def solve_transportation_problem():
    """
    Solve the Italian transportation company's empty container transport optimization problem
    Return the optimal total cost
    """
    try:
        # 1. Create model
        model = gp.Model("EmptyContainerTransport")
        
        # 2. Define data
        # Warehouse sets and supply
        warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
        I = range(len(warehouses))
        supply = {
            0: 10,  # Verona
            1: 12,  # Perugia
            2: 20,  # Rome
            3: 24,  # Pescara
            4: 18,  # Taranto
            5: 40   # Lamezia
        }
        
        # Port sets and demand
        ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]
        J = range(len(ports))
        demand = {
            0: 20,  # Genoa
            1: 15,  # Venice
            2: 25,  # Ancona
            3: 33,  # Naples
            4: 21   # Bari
        }
        
        # Distance matrix (km)
        distance = [
            # Genoa, Venice, Ancona, Naples, Bari
            [290, 115, 355, 715, 810],   # Verona
            [380, 340, 165, 380, 610],   # Perugia
            [505, 530, 285, 220, 450],   # Rome
            [655, 450, 155, 240, 315],   # Pescara
            [1010, 840, 550, 305, 95],   # Taranto
            [1072, 1097, 747, 372, 333]  # Lamezia
        ]
        
        # Unit cost (euros/km·truck)
        c = 30
        
        # 3. Create decision variables
        # x[i,j]: number of containers from warehouse i to port j
        # y[i,j]: number of trucks from warehouse i to port j
        x = model.addVars(I, J, vtype=GRB.INTEGER, name="x", lb=0)
        y = model.addVars(I, J, vtype=GRB.INTEGER, name="y", lb=0)
        
        # 4. Set objective function: minimize total transportation cost
        objective = gp.quicksum(c * distance[i][j] * y[i, j] for i in I for j in J)
        model.setObjective(objective, GRB.MINIMIZE)
        
        # 5. Add constraints
        
        # Supply constraint: each warehouse ships no more than its supply
        for i in I:
            model.addConstr(gp.quicksum(x[i, j] for j in J) <= supply[i], 
                          name=f"supply_{warehouses[i]}")
        
        # Demand constraint: each port receives exactly its demand
        for j in J:
            model.addConstr(gp.quicksum(x[i, j] for i in I) == demand[j], 
                          name=f"demand_{ports[j]}")
        
        # Truck capacity constraint: each truck can carry at most 2 containers
        for i in I:
            for j in J:
                model.addConstr(x[i, j] <= 2 * y[i, j], 
                              name=f"truck_capacity_{i}_{j}")
        
        # 6. Set solver parameters
        model.setParam('OutputFlag', 1)  # Show solving process
        model.setParam('MIPGap', 0.01)   # Set optimality gap to 1%
        
        # 7. Solve the model
        model.optimize()
        
        # 8. Output results
        if model.status == GRB.OPTIMAL:
            print(f"Optimization successful! Optimal total cost: {model.objVal:.2f} euros")
            print("\nTransportation plan details (only non-zero routes):")
            print("-" * 80)
            print(f"{'Warehouse':<10} {'Port':<10} {'Containers':<12} {'Trucks':<12} {'Distance(km)':<12} {'Cost(euros)':<12}")
            print("-" * 80)
            
            total_containers = 0
            for i in I:
                for j in J:
                    if x[i, j].x > 0:
                        containers = x[i, j].x
                        trucks = y[i, j].x
                        dist = distance[i][j]
                        cost = c * dist * trucks
                        total_containers += containers
                        print(f"{warehouses[i]:<10} {ports[j]:<10} {containers:<12.0f} {trucks:<12.0f} {dist:<12} {cost:<12.2f}")
            
            print("-" * 80)
            print(f"Total containers transported: {total_containers}")
            
            # Verify supply and demand
            print("\nSupply and demand verification:")
            for i in I:
                shipped = sum(x[i, j].x for j in J)
                print(f"{warehouses[i]}: Supply={supply[i]}, Shipped={shipped}, Remaining={supply[i]-shipped}")
            
            for j in J:
                received = sum(x[i, j].x for i in I)
                print(f"{ports[j]}: Demand={demand[j]}, Received={received}, Shortage={demand[j]-received}")
            
            return model.objVal
            
        else:
            print(f"No optimal solution found. Status code: {model.status}")
            return None
            
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        return None
    except Exception as e:
        print(f"Other error: {e}")
        return None

# Run the solving function
if __name__ == "__main__":
    optimal_cost = solve_transportation_problem()
    if optimal_cost is not None:
        print(f"\nOptimal objective value: {optimal_cost:.2f} euros")