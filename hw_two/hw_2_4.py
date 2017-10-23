import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab


np.set_printoptions(threshold='nan')
mat = scipy.io.loadmat('hw1data.mat')

data = mat['X'].astype(np.float64)
labels = mat['Y'].astype(int)

split = 7000
total = 10000
n = split

def perceptron_v0(digit):
    T = split*1
    w = [0]*784
    for t in range(1,T):
        i = (t % n + 1)
        x = data[i]
        y = -1
        if (labels[i] == digit):
            y = 1
        if (y * (np.dot(w,x)) <= 0):
            w = w + y*x
    return w

def perceptron_v1(digit):
    T = split*1
    w = [0]*784
    for t in range(1,T):
        print t
        i = 0
        product = float("inf")
        for j in range(0,split):
            yj = -1
            if (labels[j] == digit):
                yj = 1
            productTemp = yj*np.dot(w, data[j])
            if (productTemp < product):
                product = productTemp
                i = j
        y = -1
        if (labels[i] == digit):
            y = 1
        x = data[i]

        if (y * (np.dot(w,x)) <= 0):
            w = w + y*x
        else:
            break
    return w

def test_p0():
    w = [0]*10
    testSet = data[split:total]
    for digit in range(10):
        w[digit] = perceptron_v0(digit)
        print np.dot(w[digit], testSet[0])


    correctcnt = 0
    wrongcnt = 0
    for i in range(len(testSet)):
        x = testSet[i]

        guess = 0
        for digit in range(10):
            if (np.dot(w[digit], x) >= 0):
                guess = digit

        if guess == labels[i + split, 0]:
            correctcnt += 1
        else:
            wrongcnt += 1

        print "guess ", guess
        print "actual ", labels[i + split, 0]

        print correctcnt, wrongcnt
        # print w

def test_p1():
    w = [0]*10
    testSet = data[split:total]
    for digit in range(10):
        w[digit] = perceptron_v1(digit)
        print np.dot(w[digit], testSet[0])


    correctcnt = 0
    wrongcnt = 0
    for i in range(len(testSet)):
        x = testSet[i]

        guess = 0
        for digit in range(10):
            if (np.dot(w[digit], x) >= 0):
                guess = digit

        if guess == labels[i + split, 0]:
            correctcnt += 1
        else:
            wrongcnt += 1

        print "guess ", guess
        print "actual ", labels[i + split, 0]

        print correctcnt, wrongcnt
        # print w


def main():
    # test_p0()
    test_p1()

main()