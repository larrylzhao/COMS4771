import numpy
import math
import scipy.io
import matplotlib.pyplot
import time
import random

data = scipy.io.loadmat('hw2data.mat')
x = data['X']
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

# Train the neural network.
#
# Inputs x_array and y_array are vectors of size 1 ... n.
# Input k is the size of the intermediate layer.
def neural_network_iterate(x_array, y_array, k, w1, w2, b1, b2):

    # Take k inputs at a time.
    starting_indices = range(len(x_array) - k + 1)[0::k]
    for i in starting_indices:

        # Extract k inputs and labels - convert into numpy arrays.
        x = numpy.array(x_array[i:(i + k)]) # e.g. [3, 5, 7]
        y = numpy.array(y_array[i:(i + k)]) # e.g. [4, 6, 8]

        # Built network output [y_hat] given [x] and all four current weights.
        y_hat = [network_output(xi, w1, b1, w2, b2) for xi in x]

        # Compute 4 gradients and use them to update 4 weight parameters
        for j in range(len(x)):
            w1 = w1 - step * gradient_w1(x[j], y[j], w1, b1, w2, b2)
        for j in range(len(x)):
            b1 = b1 - step * gradient_b1(x[j], y[j], w1, b1, w2, b2)
        for j in range(len(x)):
            w2 = w2 - step * gradient_w2(x[j], y[j], w1, b1, w2, b2)
        for j in range(len(x)):
            b2 = b2 - step * gradient_b2(x[j], y[j], w1, b1, w2, b2)

    return w1, w2, b1, b2

# Compute the expit term
def expit(x, y, w1, b1, w2, b2):
    return (-1 * w2 / (math.exp(-1 * w1 * x - b1) + 1)) - b2

# For each of the k data points in x, compute the gradient with respect to w1.

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

neural_network(x, y, 10)

# Once converged, plot using the constants.
matplotlib.pyplot.scatter(x, y, linewidths = 0, s = 3, c = 'b')
matplotlib.pyplot.scatter(x, network_output(x, 0.61948176, -1.84151709, -2.04766945, 1.77857143), linewidths = 0, s = 3, c = 'r')
matplotlib.pyplot.show()
