# with a 70/30 training/test split, k=1, and norm L2:
# 2841 classified correctly, and 159 incorrectly,
# leading to an accuracy of 94.7%

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import operator

np.set_printoptions(threshold='nan')
mat = scipy.io.loadmat('hw1data.mat')

def knn(k, training, test):
    distances = [None]*len(training)

    for i in range(len(training)):
        d = np.linalg.norm(training[i] - test, ord=2)
        distances[i] = (d, i)
    distances.sort(key=operator.itemgetter(0))

    neighbors = [None]*k
    for i in range(k):
        neighbors[i] = distances[i][1]
    return neighbors

def main():
    data = mat['X'].astype(np.float64)
    labels = mat['Y'].astype(int)

    total = len(data)
    split = 7000
    k = 1

    training = data[0:split]
    # testSet = data[split:total]
    testSet = data[split:total]

    # look at all the test points
    correctcnt = 0
    wrongcnt = 0
    for i in range(len(testSet)):
        test = testSet[i]
        neighbors = knn(k, training, test)

        # print neighbors
        votes = [0]*10
        for neighbor in neighbors:
            digit = labels[neighbor]
            votes[digit[0]] += 1
        guess = 0
        biggest = 0
        for j in range(len(votes)):
            if votes[j] > biggest:
                guess = j
                biggest = votes[j]

        if guess == labels[i + split, 0]:
            correctcnt += 1
        else:
            wrongcnt += 1

        print "guess ", guess
        print "actual ", labels[i + split, 0]

        print correctcnt, wrongcnt


main()
