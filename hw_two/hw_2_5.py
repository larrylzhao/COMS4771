import numpy
import math
import scipy.io
import matplotlib.pyplot
import time
import random

data = scipy.io.loadmat('hw2data.mat')

# print 'The X-matrix in hw2data.mat is:'
# print data['X']
# print data['X'].shape
x = data['X']

# print 'The Y-matrix in hw2data.mat is:'
# print data['Y']
# print data['Y'].shape
y = data['Y']

# Step
step = 0.0001

# Neural Network Implementation.
#
# Inputs x_array and y_array are array vectors of size 1 ... n.
# Input k is the size of the intermediate layer.
def neural_network(x_array, y_array, k):
    # Initialize weights randomly.

    w1 = random.random()
    w2 = random.random()
    b1 = random.random()
    b2 = random.random()

    print "starting params are: "
    print '     ' + str(w1)
    print '     ' + str(w2)
    print '     ' + str(b1)
    print '     ' + str(b2)

    while True:
        new_weights = neural_network_iterate(x_array, y_array, k, w1, w2, b1, b2)
        w1 = new_weights[0]
        w2 = new_weights[1]
        b1 = new_weights[2]
        b2 = new_weights[3]
        print "params are: "
        print '     ' + str(w1)
        print '     ' + str(w2)
        print '     ' + str(b1)
        print '     ' + str(b2)
        # check for convergence



# Train the neural network, iterating through all of the data once.
#
# Inputs x and y are vectors of size 1 ... n.
# Input k is the size of the intermediate layer.
def neural_network_iterate(x_array, y_array, k, w1, w2, b1, b2):

    for i in range(len(x_array)):

        # Extract a single input and output.
        x = x_array[i]
        y = y_array[i]

        # Compute the network output y_hat given x and all four current weights.
        # If y_hat and y have the same sign (correctly classified), then skip to
        # the next data point.
        y_hat = network_output(x, w1, b1, w2, b2)
        # if (numpy.sign(y) == numpy.sign(y_hat)): continue
        if (y == y_hat): continue

        # Compute 4 gradients and use them to update 4 weight parameters
        w1 = w1 - step * gradient_w1(x, y, w1, b1, w2, b2)
        b1 = b1 - step * gradient_b1(x, y, w1, b1, w2, b2)
        w2 = w2 - step * gradient_w2(x, y, w1, b1, w2, b2)
        b2 = b2 - step * gradient_b2(x, y, w1, b1, w2, b2)

    return w1, w2, b1, b2

# Compute the expit term
def expit(x, y, w1, b1, w2, b2):
    return (-1 * w2 / (math.exp(-1 * w1 * x - b1) + 1)) - b2

# Compute the gradient with respect to w1 at the data point (x) with label (y).
def gradient_w1(x, y, w1, b1, w2, b2):
    top_1 = 2 * w2 * x * math.exp(expit(x, y, w1, b1, w2, b2) - w1 * x - b1)
    top_2 = 1 / (math.exp(expit(x, y, w1, b1, w2, b2)) + 1) - y
    bottom_1 = math.pow(math.exp(-1 * w1 * x - b1) + 1, 2)
    bottom_2 = math.pow(math.exp(expit(x, y, w1, b1, w2, b2)) + 1, 2)
    return (top_1 * top_2) / (bottom_1 * bottom_2)

# Compute the gradient with respect to b1 at the data point (x) with label (y).
def gradient_b1(x, y, w1, b1, w2, b2):
    return gradient_w1(x, y, w1, b1, w2, b2) / x

# Compute the gradient with respect to w2 at the data point (x) with label (y).
def gradient_w2(x, y, w1, b1, w2, b2):
    top_1 = 2 * math.exp(expit(x, y, w1, b1, w2, b2))
    top_2 = 1 / (math.exp(expit(x, y, w1, b1, w2, b2)) + 1) - y
    bottom_1 = math.exp(-1 * w1 * x - b1) + 1
    bottom_2 = math.pow(math.exp(expit(x, y, w1, b1, w2, b2)) + 1, 2)
    return (top_1 * top_2) / (bottom_1 * bottom_2)

# Compute the gradient with respect to b2 at the data point (x) with label (y).
def gradient_b2(x, y, w1, b1, w2, b2):
    top_1 = 2 * math.exp(expit(x, y, w1, b1, w2, b2))
    top_2 = 1 / (math.exp(expit(x, y, w1, b1, w2, b2)) + 1) - y
    bottom = math.pow(math.exp(expit(x, y, w1, b1, w2, b2)) + 1, 2)
    return top_1 * top_2 / bottom

# Compute the expit function value at x, 1/1+e^(-wx-b), with the specified weights.
def F_value(x, w, b):
    return scipy.special.expit(w * x + b)

# Compute the error for a single data point given the current weights.
# The error is the square of the difference between network output (predicted
# label) and the actual label.
def E_value(x, y, w1, b1, w2, b2):
    return (network_output(x, w1, b1, w2, b2) - y) ** 2

# Compute the network output for a single data point given the current weights.
def network_output(x, w1, b1, w2, b2):
    return F_value(F_value(x, w1, b1), w2, b2)

# params are:
#      [ 0.55251848]
#      [-1.86505591]
#      [-1.66352657]
#      [ 1.80538865]

neural_network(x, y, 2)

matplotlib.pyplot.scatter(x, y, linewidths = 0, s = 3, c = 'b')
matplotlib.pyplot.scatter(x, network_output(x, 0.5525, -1.8651, -1.6635, 1.8054), linewidths = 0, s = 3, c = 'r')
matplotlib.pyplot.show()
