# Essentially calculate the k closest distances
# do this by subtracting training point array from test point array
# square the differences and add them all up
# then take the square root of that sum


import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import operator

np.set_printoptions(threshold='nan')
mat = scipy.io.loadmat('hw1data.mat')

def distance(x, y):
    d = 0
    for i in range(len(x)):
        d += pow((x[i] - y[i]), 2)
    d = math.sqrt(d)
    return d

def knn(k, training, test):
    distances = [None]*len(training)

    for i in range(len(training)):
        d = distance(training[i], test)
        distances[i] = (d, training[i])
    distances.sort(key=operator.itemgetter(0))

    neighbors = [None]*k
    for i in range(k):
        neighbors[i] = distances[i][1]
    return neighbors

def main():
    data = mat['X'].astype(np.float64)
    labels = mat['Y'].astype(np.float64)

    total = len(data)
    split = 7000
    k = 5

    training = data[0:split]
    # testSet = data[split:total]
    testSet = data[split:split+1]

    # look at all the test points
    for test in testSet:
        neighbors = knn(k, training, test)

        print neighbors
        votes = [0]*10

main()
