import gurobipy as gp
from gurobipy import GRB

# Data definitions
warehouses = ['Verona', 'Perugia', 'Rome', 'Pescara', 'Taranto', 'Lamezia']
ports = ['Genoa', 'Venice', 'Ancona', 'Naples', 'Bari']

supply = {
    'Verona': 10,
    'Perugia': 12,
    'Rome': 20,
    'Pescara': 24,
    'Taranto': 18,
    'Lamezia': 40
}

demand = {
    'Genoa': 20,
    'Venice': 15,
    'Ancona': 25,
    'Naples': 33,
    'Bari': 21
}

# Distance matrix in kilometers
distance = {
    'Verona': {'Genoa': 290, 'Venice': 115, 'Ancona': 355, 'Naples': 715, 'Bari': 810},
    'Perugia': {'Genoa': 380, 'Venice': 340, 'Ancona': 165, 'Naples': 380, 'Bari': 610},
    'Rome': {'Genoa': 505, 'Venice': 530, 'Ancona': 285, 'Naples': 220, 'Bari': 450},
    'Pescara': {'Genoa': 655, 'Venice': 450, 'Ancona': 155, 'Naples': 240, 'Bari': 315},
    'Taranto': {'Genoa': 1010, 'Venice': 840, 'Ancona': 550, 'Naples': 305, 'Bari': 95},
    'Lamezia': {'Genoa': 1072, 'Venice': 1097, 'Ancona': 747, 'Naples': 372, 'Bari': 333}
}

try:
    # Create optimization model
    model = gp.Model('ContainerTransportation')
    
    # Create decision variables: number of containers shipped from warehouse i to port j
    x = {}
    for warehouse in warehouses:
        for port in ports:
            x[warehouse, port] = model.addVar(
                lb=0.0, 
                vtype=GRB.CONTINUOUS, 
                name=f'ship_{warehouse}_{port}'
            )
    
    # Set objective: minimize total transportation cost
    # Cost per container = 30 * distance
    objective_expr = gp.quicksum(
        30.0 * distance[warehouse][port] * x[warehouse, port] 
        for warehouse in warehouses 
        for port in ports
    )
    model.setObjective(objective_expr, GRB.MINIMIZE)
    
    # Add supply constraints: total shipments from each warehouse <= available containers
    for warehouse in warehouses:
        supply_constraint = gp.quicksum(
            x[warehouse, port] for port in ports
        ) <= supply[warehouse]
        model.addConstr(supply_constraint, name=f'supply_{warehouse}')
    
    # Add demand constraints: total shipments to each port = exact demand
    for port in ports:
        demand_constraint = gp.quicksum(
            x[warehouse, port] for warehouse in warehouses
        ) == demand[port]
        model.addConstr(demand_constraint, name=f'demand_{port}')
    
    # Solve the optimization model
    model.optimize()
    
    # Check and display solution status
    if model.status == GRB.OPTIMAL:
        print(f'Optimal transportation cost: {model.objVal:.2f} euros')
        print('\nDetailed shipment plan (non-zero routes):')
        print('-' * 40)
        
        # Display non-zero shipments
        for warehouse in warehouses:
            for port in ports:
                flow = x[warehouse, port].x
                if flow > 0.001:  # Display significant flows
                    route_cost = 30.0 * distance[warehouse][port] * flow
                    print(f'{warehouse:8s} -> {port:7s}: {flow:3.0f} containers, '
                          f'Cost: {route_cost:8.2f} euros')
        
        print('\nSummary statistics:')
        print('-' * 40)
        
        # Calculate total containers shipped
        total_containers = sum(x[warehouse, port].x 
                               for warehouse in warehouses 
                               for port in ports)
        
        # Calculate supply utilization
        print(f'Total containers shipped: {total_containers:.0f}')
        print(f'Total supply available: {sum(supply.values())}')
        print(f'Total demand satisfied: {sum(demand.values())}')
        
        # Calculate per-warehouse utilization
        print('\nWarehouse utilization:')
        for warehouse in warehouses:
            shipped = sum(x[warehouse, port].x for port in ports)
            utilization = (shipped / supply[warehouse]) * 100
            print(f'{warehouse:8s}: {shipped:3.0f}/{supply[warehouse]:2d} containers '
                  f'({utilization:5.1f}%)')
        
        # Calculate per-port sourcing
        print('\nPort sourcing:')
        for port in ports:
            received = sum(x[warehouse, port].x for warehouse in warehouses)
            print(f'{port:7s}: {received:3.0f}/{demand[port]:2d} containers')
            
    elif model.status == GRB.INFEASIBLE:
        print('Model is infeasible - no solution satisfies all constraints.')
        print('Computing Irreducible Inconsistent Subsystem (IIS)...')
        model.computeIIS()
        model.write('infeasible_model.ilp')
        print('IIS written to file: infeasible_model.ilp')
        
    elif model.status == GRB.UNBOUNDED:
        print('Model is unbounded - objective can be improved indefinitely.')
        
    elif model.status == GRB.INF_OR_UNBD:
        print('Model is infeasible or unbounded.')
        
    elif model.status == GRB.TIME_LIMIT:
        print('Optimization stopped due to time limit.')
        if model.solCount > 0:
            print(f'Best solution found: {model.objVal:.2f} euros')
            
    else:
        print(f'Optimization terminated with status code: {model.status}')

except gp.GurobiError as e:
    print(f'Gurobi optimization error: {e}')
    
except KeyError as e:
    print(f'Data access error: Missing key {e}')
    
except Exception as e:
    print(f'Unexpected error: {e}')