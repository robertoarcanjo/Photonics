"""
Implementation of numerical methods
"""

import numpy as np

class NumericalMethods:

    def __init__(self, y0, k0, k1, g):
        self.y0 = y0
        self.k0 = k0
        self.k1 = k1
        self.g = g

    """
    Methods to solve ODEs utilizing eq. (4):

    1-solve_ode_eq_4

    To solve an ODE utilizing eq. (4), we need pass the following parameters:
    - The step size h
    - The initial condition y(t0) = y0
    - The time span [t0, tf]
    - The function y'(x)
    

    function:

    Assuming we have an ODE function defined as follows:

    k0*f'(x) + k1*f(x) = g(x)

    f'(x) = (g(x) - k1*f(x)) / k0

    f(x+h)-f(x-h)/2h = f'(x)

    f(x+h) = ((g(x) - k1*f(x)) / k0)*2h + f(x-h)

    f(x) is a mean value between f(x+h) and f(x-h)

    f(x) = (g(x/2) - f(x-h))*h/2 + f(x-h)

    #Example: Defining g:
    g = lambda x: np.sin(x)

    """

    def solve_ode_eq_4(self, h, t_span):
        t0, tf = t_span
        N = int((tf - t0) / h)
        t_values = np.linspace(t0, tf, N + 1)
        y_values = np.zeros(N + 1)
        y_values[0] = self.y0

        for i in range(1, N + 1):
            mean = ((self.g(t_values[i]/2) - y_values[i-1]) * 0.5 * h) + y_values[i-1]
            y_values[i] = ((self.g(t_values[i]) - self.k1 * mean) / self.k0) * (h) + y_values[i-1]

        return t_values, y_values
    
