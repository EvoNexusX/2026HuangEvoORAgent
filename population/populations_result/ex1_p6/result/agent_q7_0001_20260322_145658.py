from gurobipy import Model, GRB

model = Model("EmptyContainerTransportation")

warehouses = ['Verona', 'Perugia', 'Rome', 'Pescara', 'Taranto', 'Lamezia']
ports = ['Genoa', 'Venice', 'Ancona', 'Naples', 'Bari']

supply = {'Verona': 10, 'Perugia': 12, 'Rome': 20, 'Pescara': 24, 'Taranto': 18, 'Lamezia': 40}
demand = {'Genoa': 20, 'Venice': 15, 'Ancona': 25, 'Naples': 33, 'Bari': 21}

cost_per_km = 30

distance = {
    ('Verona', 'Genoa'): 290, ('Verona', 'Venice'): 115, ('Verona', 'Ancona'): 355,
    ('Verona', 'Naples'): 715, ('Verona', 'Bari'): 810,
    ('Perugia', 'Genoa'): 380, ('Perugia', 'Venice'): 340, ('Perugia', 'Ancona'): 165,
    ('Perugia', 'Naples'): 380, ('Perugia', 'Bari'): 610,
    ('Rome', 'Genoa'): 505, ('Rome', 'Venice'): 530, ('Rome', 'Ancona'): 285,
    ('Rome', 'Naples'): 220, ('Rome', 'Bari'): 450,
    ('Pescara', 'Genoa'): 655, ('Pescara', 'Venice'): 450, ('Pescara', 'Ancona'): 155,
    ('Pescara', 'Naples'): 240, ('Pescara', 'Bari'): 315,
    ('Taranto', 'Genoa'): 1010, ('Taranto', 'Venice'): 840, ('Taranto', 'Ancona'): 550,
    ('Taranto', 'Naples'): 305, ('Taranto', 'Bari'): 95,
    ('Lamezia', 'Genoa'): 1072, ('Lamezia', 'Venice'): 1097, ('Lamezia', 'Ancona'): 747,
    ('Lamezia', 'Naples'): 372, ('Lamezia', 'Bari'): 333
}

x = model.addVars(warehouses, ports, name="x")

model.setObjective(
    sum(cost_per_km * distance[i, j] * x[i, j] for i in warehouses for j in ports),
    GRB.MINIMIZE
)

model.addConstrs(
    (sum(x[i, j] for j in ports) <= supply[i] for i in warehouses),
    name="supply"
)

model.addConstrs(
    (sum(x[i, j] for i in warehouses) == demand[j] for j in ports),
    name="demand"
)

model.optimize()

if model.status == GRB.OPTIMAL:
    print(f'最优总成本: {model.objVal:.2f} 欧元')
    for i in warehouses:
        for j in ports:
            if x[i, j].x > 1e-6:
                print(f'{i} -> {j}: {x[i, j].x:.1f}')
else:
    print("未找到最优解")