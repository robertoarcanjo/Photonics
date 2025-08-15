import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def U_v2(Vo,x,y,n,a,b):
    U = 0

    for i in range(1, n+1, 2):
        U += (4*Vo/((i)*np.pi))*(1/np.sinh((i)*np.pi*a/b))*np.sinh((i)*np.pi*x/b)*np.sin((i)*np.pi*y/b)
    return U

a = 2
b = 2
Vo = 1

X = np.arange(0, a, 0.1)
Y = np.arange(0, b, 0.1)
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