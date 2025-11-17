# libraries ----------------------------------------------
import numpy as np
import json
# --------------------------------------------------------

"""
This class currently supports only 2D simulations 
for a few specific waveguide structures.
"""

class NumericalSolver:

    def __init__(self, jsonPath):
        self.micro_m = 1e-6
        self.params = self.loadParams(jsonPath)
        pass

    def loadParams(self, jsonPath):
        with open(jsonPath, 'r') as f:
            params = json.load(f)
        """self.wavelength = params["wavelength"] * self.micro_m
        self.n_clad = params["n_clad"]
        self.n_core = params["n_core"]
        self.core_width = params["core_width"] * self.micro_m
        self.core_height = params["core_height"] * self.micro_m
        self.grid_size = params["grid_size"] * self.micro_m"""
        return params
    
    