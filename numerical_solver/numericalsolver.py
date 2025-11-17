# libraries ----------------------------------------------
import numpy as np

import json

import h5py

import matplotlib.pyplot as plt

import scipy.sparse as sp
# --------------------------------------------------------

"""
This class currently supports only 2D simulations 
for a few specific waveguide structures.
"""

class NumericalSolver:

    def __init__(self, jsonPath):
        self.micro_m = 1e-6
        self.params = self.loadParams(jsonPath)
        #self.spacer = 1*self.params['basic-params']['wavelength']
        self.spacer = 2.6e-6

    def loadParams(self, jsonPath):
        with open(jsonPath, 'r') as f:
            params = json.load(f)
        return params
    
    def createGrid(self):

        N_max = max(self.params['basic-params']['eta_c'], 
                    self.params['basic-params']['eta_f'], 
                    self.params['basic-params']['eta_s'])

        dx = self.params['basic-params']['wavelength'] / (N_max * self.params['grid']['N'])
        dy = dx

        nx = int(np.ceil(self.spacer / dx))
        ny = int(np.ceil(self.params['structure']['height-2a'] / dy))

        dx = self.spacer / nx
        dy = self.params['structure']['height-2a'] / ny

        total_length_y = self.spacer + self.params['structure']['height-2a'] + self.spacer
        total_length_x = 5*self.spacer

        Ny = int(np.ceil(total_length_y / dy))
        Nx = int(np.ceil(total_length_x / dx))

        total_length_y = Ny * dy
        total_length_x = Nx * dx

        Nx2 = 2 * Nx
        Ny2 = 2 * Ny
        dx2 = dx/2
        dy2 = dy/2

        x = np.arange(1, Nx+1) * dx
        y = np.arange(1, Ny+1) * dy

        x = x - np.mean(x)
        y = y - np.mean(y)

        x2 = np.arange(1, Nx2+1) * dx2
        y2 = np.arange(1, Ny2+1) * dy2

        x2 = x2 - np.mean(x2)
        y2 = y2 - np.mean(y2)

        X, Y = np.meshgrid(x2, y2, indexing='xy')

        grid = {
            'dx': dx,
            'dy': dy,
            'nx': nx,
            'ny': ny,
            'Nx': Nx,
            'Ny': Ny,
            'x': x,
            'y': y,
            'x2': x2,
            'y2': y2,
            'X': X,
            'Y': Y
        }

        return grid
        
    def showGrid(self, grid):
        print("Grid parameters:")
        print("dx: ", grid['dx'])
        print("dy: ", grid['dy'])
        print("nx: ", grid['nx'])
        print("ny: ", grid['ny'])
        print("Nx: ", grid['Nx'])
        print("Ny: ", grid['Ny'])

        plt.figure(figsize=(8, 6))
        plt.scatter(grid['X'], grid['Y'], s=1)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Grid Points')
        plt.grid(True)
        plt.show()
        return
    
    def waveguideGeometry(self, grid):
        if self.params['structure']['type'] == 'slab':
            eta_c = self.params['basic-params']['eta_c']
            eta_f = self.params['basic-params']['eta_f']
            eta_s = self.params['basic-params']['eta_s']
            height_2a = self.params['structure']['height-2a']

            Y = grid['Y']
            X = grid['X']

            eta = np.ones_like(X)

            core_start_y = -height_2a / 2
            core_end_y = height_2a / 2

            eta[(Y >= core_start_y) & (Y <= core_end_y)] = eta_f
            eta[Y < core_start_y] = eta_s
            eta[Y > core_end_y] = eta_c

        return eta

    def showWaveguideGeometry(self, eta):
        plt.figure(figsize=(8, 6))
        plt.imshow(eta, extent=[-self.spacer/2, self.spacer/2, -self.spacer/2, self.spacer/2], origin='lower')
        plt.colorbar(label='Eta')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Waveguide Geometry (Eta)')
        plt.grid(False)
        plt.show()
        return
    
    def diag_from_grid(self, G, grid):
        # Use o tamanho real do array G, não do grid completo
        M = G.size
        return sp.diags(G.reshape(-1, order='F'), 0, shape=(M,M), format='csr')
    
    def diff_opr(self, Nx,Ny,dx,dy):
        M = Nx*Ny
        diagonal = np.ones(M-1)
        diagonal[Nx-1:-1:Nx] = 0
        Dxf = -sp.eye(M,M) + sp.diags(diagonal, 1, shape=(M,M))
        Dyf = sp.eye(M,M,k = Nx) - sp.eye(M,M)
        Dxg = -Dxf.conj().T
        Dyg = -Dyf.conj().T
        return Dxf/dx, Dyf/dy, Dxg/dx, Dyg/dy
    
    def inv_diag(self, diag_mat: sp.csr_matrix) -> sp.csr_matrix:
        d = diag_mat.diagonal()
        return sp.diags(1.0 / d,format='csr')
    
    def computeEquations(self, grid, eta):

        eps_r = eta**2
        mu_r = np.ones_like(eta)

        eps_r_x = eps_r[1::2, 0::2]
        eps_r_y = eps_r[0::2, 1::2]
        eps_r_z = eps_r[0::2, 0::2]

        mu_r_x = mu_r[0::2, 1::2]
        mu_r_y = mu_r[1::2, 0::2]
        mu_r_z = mu_r[1::2, 1::2]

        M = eps_r_x.size
        assert M == (grid['Ny']) * (grid['Nx']), "M size mismatch"

        eps_r_x_diag = self.diag_from_grid(eps_r_x, grid)
        eps_r_y_diag = self.diag_from_grid(eps_r_y, grid)
        eps_r_z_diag = self.diag_from_grid(eps_r_z, grid)

        mu_r_x_diag = self.diag_from_grid(mu_r_x, grid)
        mu_r_y_diag = self.diag_from_grid(mu_r_y, grid)
        mu_r_z_diag = self.diag_from_grid(mu_r_z, grid)

        K0 = 2 * np.pi / self.params['basic-params']['wavelength']

        DEX, DEY, DHX, DHY = self.diff_opr(grid['Nx'], grid['Ny'], grid['dx'], grid['dy'])

        DEX = DEX/K0
        DEY = DEY/K0
        DHX = DHX/K0
        DHY = DHY/K0

        eps_r_z_inv = self.inv_diag(eps_r_z_diag)
        mu_r_z_inv = self.inv_diag(mu_r_z_diag)

        P11 = DEX @ eps_r_z_inv @ DHY
        P12 = - (DEX @ eps_r_z_inv @ DHX + mu_r_y_diag)
        P21 = DEY @ eps_r_z_inv @ DHY + mu_r_x_diag
        P22 = - (DEY @ eps_r_z_inv @ DHX)

        Q11 = DHX @ mu_r_z_inv @ DEY
        Q12 = - (DHX @ mu_r_z_inv @ DEX + eps_r_y_diag)
        Q21 =  (DHY @ mu_r_z_inv @ DEY + eps_r_x_diag)
        Q22 = - (DHY @ mu_r_z_inv @ DEX)

        P = sp.bmat([[P11, P12],
                         [P21, P22]], format='csr')
        
        Q = sp.bmat([[Q11, Q12],
                         [Q21, Q22]], format='csr')
        
        A = (P @ Q).tocsr()
        ev_target = -(self.params['basic-params']['eta_f']**2)

        vals, vecs = sp.linalg.eigs(A, k=self.params['basic-params']['num-modes'], sigma=ev_target)

        d = np.sqrt(vals)
        eta_eff = -1j*d

        Ex = vecs[0:M, :]

        return eta_eff, Ex
    
    def solve(self):

        if self.params["solver"]["type"] == "yee-algorithm":
            print("Yee-algorithm selected")

        

        grid = self.createGrid()
        eta = self.waveguideGeometry(grid)
        eta_eff, Ex = self.computeEquations(grid, eta)

        return eta_eff, Ex
    
    def saveResults(self, eta_eff, Ex, h5Path):
        with h5py.File(h5Path, 'w') as f:
            f.create_dataset('eta_eff', data=eta_eff)
            f.create_dataset('Ex', data=Ex)
        return
    
    def loadResults(self, h5Path):
        with h5py.File(h5Path, 'r') as f:
            eta_eff = f['eta_eff'][:]
            Ex = f['Ex'][:]
        return eta_eff, Ex
    
    def plotModes(self, Ex, grid, mode_indices):
        
        Nx = grid['Nx']
        Ny = grid['Ny']

        for idx in mode_indices:
            Ex_mode = Ex[:, idx]
            Ex_mode_2D = np.reshape(Ex_mode, (Nx, Ny), order='F')

            plt.figure(figsize=(8, 6))
            plt.imshow(np.real(Ex_mode_2D.T), extent=np.array([grid['x'].min(), grid['x'].max(), grid['y'].max(), grid['y'].min()]), aspect='equal')
            plt.colorbar(label='|Ex|')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title(f'Mode {idx} - |Ex| Distribution')
            plt.grid(False)
            plt.show()
        return