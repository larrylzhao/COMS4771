import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import random

# (i)
L = np.matrix([[1.25, -1.5],
     [-1.5, 5]])

print L

# (ii)
R = [None]*500
for i in range (0, 500):
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)
    slope = y/x
    x = (1+slope**2)**(-.5)
    y = slope*x
    R[i] = np.matrix([[x], [y]])

# (iii)
Rdistort = [None]*500
for i in range (0, 500):
    r = R[i]
    Rdistort[i] = np.dot(L, r)

# (iv)
w, v = np.linalg.eig(L)
w.sort()
eigMax = w[-1] #5.52617158904
eigMin = w[0] # 0.723828410963

# (v)
Rlength = [None]*500
for i in range (0, 500):
    r = Rdistort[i]
    x = float(r[0][0])
    y = float(r[1][0])
    Rlength[i] = (x**2 + y**2)**(.5)

# (vi)
binSize = 50
# n, bins, patches = plt.hist(Rlength, binSize, facecolor='blue', alpha=0.5)
# plt.show()

# (vii)

# (viii)
v.sort()
vMax = v[:,-1]
# [[ 0.33100694]
#  [-0.33100694]]

# (ix)
X = [0]*500
Y = [0]*500
U = [None]*500
V = [None]*500
for i in range (0, 500):
    U[i] = float(Rdistort[i][0])
    V[i] = float(Rdistort[i][1])

ax = plt.gca()
ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='blue')
X = [0]
Y = [0]
U = [float(vMax[0])]
V = [float(vMax[1])]
ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='red')
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
plt.draw()
plt.show()

# (x)

