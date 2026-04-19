# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

def main():
    # Data definition
    warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
    ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]
    
    I = range(len(warehouses))  # Warehouse indices
    J = range(len(ports))       # Port indices
    
    s = [10, 12, 20, 24, 18, 40]  # Supply
    d = [20, 15, 25, 33, 21]      # Demand
    
    # Distance matrix (km)
    dist = [
        [290, 115, 355, 715, 810],
        [380, 340, 165, 380, 610],
        [505, 530, 285, 220, 450],
        [655, 450, 155, 240, 315],
        [1010, 840, 550, 305, 95],
        [1072, 1097, 747, 372, 333]
    ]
    
    # Create model
    model = gp.Model("ContainerTransport")
    
    # Define variables
    x = model.addVars(I, J, lb=0.0, name="x")
    
    # Set objective: minimize total transportation cost
    model.setObjective(
        gp.quicksum(30 * dist[i][j] * x[i, j] for i in I for j in J),
        GRB.MINIMIZE
    )
    
    # Add supply constraints
    for i in I:
        model.addConstr(
            gp.quicksum(x[i, j] for j in J) <= s[i],
            name=f"Supply_{warehouses[i]}"
        )
    
    # Add demand constraints
    for j in J:
        model.addConstr(
            gp.quicksum(x[i, j] for i in I) == d[j],
            name=f"Demand_{ports[j]}"
        )
    
    # Solve the model
    model.optimize()
    
    # Output results
    if model.status == GRB.OPTIMAL:
        print(f"Minimum total cost: {model.objVal:.2f} euros")
        print("\nTransportation plan (only showing routes with positive flow):")
        for i in I:
            for j in J:
                if x[i, j].x > 1e-6:
                    print(f"  From {warehouses[i]} to {ports[j]}: {x[i, j].x:.0f} containers")
        
        print("\nRemaining containers at each warehouse:")
        for i in I:
            shipped = sum(x[i, j].x for j in J)
            remaining = s[i] - shipped
            print(f"  {warehouses[i]}: {remaining:.0f}")
    else:
        print("No optimal solution found")

if __name__ == "__main__":
    main()