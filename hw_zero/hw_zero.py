import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

mat = scipy.io.loadmat('hw0data.mat')

print mat['M']
matrix = mat['M']

dimensions = matrix.shape
print "(ii) " + str(dimensions)

print "(iii) " + str(matrix[3][4])

total = 0
for row in range (0, dimensions[0]):
    total = total + matrix[row][4]
mean = float(total) / float(dimensions[0])
print "(iv) " + str(mean)

n, bins, patches = plt.hist(matrix[3], facecolor='blue', alpha=0.5)
plt.show()

m = np.matrix(matrix.astype(int))
mT = m.T
product = np.dot(mT, m)
w, v = np.linalg.eig(product)
w.sort()
w = w[::-1]
five = "(vi) "
for i in range (0, 3):
    five = five + str(w[i])
    if i != 2:
        five = five + ", "
print five