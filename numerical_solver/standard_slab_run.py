import numericalsolver as ns

slab = ns.NumericalSolver("numerical_solver/input_problems/standard_slab.json")
eta_eff, Ex = slab.solve()
grid = slab.createGrid()
eta = slab.waveguideGeometry(grid)
slab.showWaveguideGeometry(eta)
slab.plotModes(Ex, slab.createGrid(), mode_indices=[0,1,2])