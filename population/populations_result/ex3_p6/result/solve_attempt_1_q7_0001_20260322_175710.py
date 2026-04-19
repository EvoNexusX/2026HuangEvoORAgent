import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_transportation_problem():
    """
    Solves the container transportation problem to minimize total shipping costs.
    Uses a linear programming model with the transportation problem structure.
    """
    try:
        # Create optimization model
        model = gp.Model("ItalianContainerTransport")
        model.setParam('OutputFlag', 1)  # Show optimization progress
        
        # ===== DATA DEFINITION =====
        # Warehouse data (supply points)
        warehouses = ['Verona', 'Perugia', 'Rome', 'Pescara', 'Taranto', 'Lamezia']
        supply = {
            'Verona': 10,
            'Perugia': 12,
            'Rome': 20,
            'Pescara': 24,
            'Taranto': 18,
            'Lamezia': 40
        }
        
        # Port data (demand points)
        ports = ['Genoa', 'Venice', 'Ancona', 'Naples', 'Bari']
        demand = {
            'Genoa': 20,
            'Venice': 15,
            'Ancona': 25,
            'Naples': 33,
            'Bari': 21
        }
        
        # Distance matrix in kilometers (warehouse → port)
        distances = {
            'Verona': {'Genoa': 290, 'Venice': 115, 'Ancona': 355, 'Naples': 715, 'Bari': 810},
            'Perugia': {'Genoa': 380, 'Venice': 340, 'Ancona': 165, 'Naples': 380, 'Bari': 610},
            'Rome': {'Genoa': 505, 'Venice': 530, 'Ancona': 285, 'Naples': 220, 'Bari': 450},
            'Pescara': {'Genoa': 655, 'Venice': 450, 'Ancona': 155, 'Naples': 240, 'Bari': 315},
            'Taranto': {'Genoa': 1010, 'Venice': 840, 'Ancona': 550, 'Naples': 305, 'Bari': 95},
            'Lamezia': {'Genoa': 1072, 'Venice': 1097, 'Ancona': 747, 'Naples': 372, 'Bari': 333}
        }
        
        # Cost parameters
        cost_per_km = 30  # euros per kilometer per container
        
        # Validate total supply vs demand
        total_supply = sum(supply.values())
        total_demand = sum(demand.values())
        print(f"Total supply: {total_supply} containers")
        print(f"Total demand: {total_demand} containers")
        
        if total_supply < total_demand:
            print("Warning: Insufficient supply to meet all demand")
        elif total_supply > total_demand:
            print(f"Note: Excess supply of {total_supply - total_demand} containers will remain unused")
        
        # ===== DECISION VARIABLES =====
        # x[w, p] = number of containers transported from warehouse w to port p
        # Note: Using continuous variables is sufficient due to total unimodularity
        x = model.addVars(warehouses, ports,
                         vtype=GRB.CONTINUOUS,
                         name="shipment",
                         lb=0.0)
        
        # ===== OBJECTIVE FUNCTION =====
        # Minimize total transportation cost
        transportation_cost = gp.quicksum(
            cost_per_km * distances[w][p] * x[w, p]
            for w in warehouses
            for p in ports
        )
        model.setObjective(transportation_cost, GRB.MINIMIZE)
        
        # ===== CONSTRAINTS =====
        # Supply constraints: Cannot exceed available containers at each warehouse
        supply_constrs = {}
        for w in warehouses:
            constr_name = f"supply_{w}"
            supply_constrs[w] = model.addConstr(
                gp.quicksum(x[w, p] for p in ports) <= supply[w],
                name=constr_name
            )
        
        # Demand constraints: Must exactly meet port requirements
        demand_constrs = {}
        for p in ports:
            constr_name = f"demand_{p}"
            demand_constrs[p] = model.addConstr(
                gp.quicksum(x[w, p] for w in warehouses) == demand[p],
                name=constr_name
            )
        
        # ===== SOLVE THE MODEL =====
        model.optimize()
        
        # ===== SOLUTION ANALYSIS =====
        if model.status == GRB.OPTIMAL:
            # Display optimal objective value
            print(f"\n{'='*60}")
            print(f"OPTIMAL SOLUTION FOUND")
            print(f"{'='*60}")
            print(f"Minimum total transportation cost: €{model.objVal:,.2f}")
            
            # Calculate and display shipment details
            print(f"\n{'='*80}")
            print(f"OPTIMAL TRANSPORTATION PLAN")
            print(f"{'='*80}")
            print(f"{'From':<12} {'To':<12} {'Containers':<12} {'Distance':<12} {'Route Cost':<12}")
            print(f"{'-'*80}")
            
            total_containers = 0
            solution_matrix = {}
            
            for w in warehouses:
                for p in ports:
                    flow = x[w, p].x
                    if flow > 0.001:  # Display only positive flows
                        distance = distances[w][p]
                        route_cost = cost_per_km * distance * flow
                        print(f"{w:<12} {p:<12} {flow:<12.0f} {distance:<12} €{route_cost:<12.2f}")
                        total_containers += flow
                        solution_matrix[(w, p)] = flow
            
            print(f"{'='*80}")
            print(f"Total containers shipped: {total_containers:.0f}")
            
            # Display supply utilization
            print(f"\n{'='*50}")
            print(f"WAREHOUSE UTILIZATION")
            print(f"{'='*50}")
            for w in warehouses:
                shipped = sum(x[w, p].x for p in ports)
                utilization = (shipped / supply[w]) * 100 if supply[w] > 0 else 0
                print(f"{w:<12}: {shipped:>3.0f}/{supply[w]:<3} containers ({utilization:>6.1f}%)")
            
            # Display demand satisfaction
            print(f"\n{'='*50}")
            print(f"DEMAND SATISFACTION")
            print(f"{'='*50}")
            for p in ports:
                received = sum(x[w, p].x for w in warehouses)
                print(f"{p:<12}: {received:>3.0f}/{demand[p]:<3} containers")
            
            # Display cost distribution
            print(f"\n{'='*60}")
            print(f"COST DISTRIBUTION BY ROUTE")
            print(f"{'='*60}")
            
            route_costs = []
            for (w, p), flow in solution_matrix.items():
                if flow > 0:
                    cost = cost_per_km * distances[w][p] * flow
                    route_costs.append(((w, p), cost))
            
            # Sort by cost (descending)
            route_costs.sort(key=lambda item: item[1], reverse=True)
            
            for (w, p), cost in route_costs:
                percentage = (cost / model.objVal) * 100
                print(f"{w:>10} → {p:<10}: €{cost:>10,.2f} ({percentage:>6.1f}%)")
            
            return {
                'status': 'OPTIMAL',
                'total_cost': model.objVal,
                'solution_matrix': solution_matrix,
                'model': model
            }
            
        elif model.status == GRB.INFEASIBLE:
            print("\nModel is infeasible. Computing Irreducible Inconsistent Subsystem (IIS)...")
            model.computeIIS()
            model.write("infeasible_model.ilp")
            print("IIS written to file 'infeasible_model.ilp'")
            return {'status': 'INFEASIBLE'}
            
        elif model.status == GRB.UNBOUNDED:
            print("\nModel is unbounded")
            return {'status': 'UNBOUNDED'}
            
        else:
            print(f"\nOptimization terminated with status code: {model.status}")
            return {'status': 'OTHER'}
            
    except gp.GurobiError as e:
        print(f"Gurobi error occurred: {e}")
        return {'status': 'ERROR', 'message': str(e)}
        
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        return {'status': 'ERROR', 'message': str(e)}

# Main execution
if __name__ == "__main__":
    print("Italian Container Transportation Problem")
    print("=" * 50)
    print("Finding optimal transportation plan to minimize costs...\n")
    
    result = solve_transportation_problem()
    
    if result.get('status') == 'OPTIMAL':
        print("\n" + "="*60)
        print("Optimization completed successfully.")
        print("="*60)