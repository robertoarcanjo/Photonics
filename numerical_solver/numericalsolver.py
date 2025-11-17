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
        return params
    
    def waveguideGeometry(self):

        return
    
    def updateGrid(self):

        return
    
    def computeEquations(self):

        return
    
    def solve(self):

        if self.params["solver"]["type"] == "yee-algorithm":
            print("Yee-algorithm selected")
            
        else:
            pass

        return
    
