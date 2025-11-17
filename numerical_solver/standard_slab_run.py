import numericalsolver as ns

slab = ns.NumericalSolver("numerical_solver/input_problems/standard_slab.json")
slab.solve()