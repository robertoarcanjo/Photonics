import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def U_v2(Vo,x,y,n,a,b):
    U = 0

    for i in range(2, n):
        U += (4*Vo/((2*i)*np.pi))*(1/np.sinh((2*i)*np.pi*a/b))*np.sinh((2*i)*np.pi*x/b)*np.sin((2*i)*np.pi*y/b)
    return U

a = 2
b = 2
Vo = 1

X = np.arange(0, 2, 0.1)
Y = np.arange(0, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = U_v2(Vo, X, Y, 100, a, b)

# Plot the surface
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z)

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('U')

plt.show()

# Plot in 2d
fig, ax = plt.subplots()
CS = ax.contourf(X, Y, Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Contour plot of U')
plt.colorbar(CS)
plt.show()