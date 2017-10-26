import scipy.io
import numpy as np


np.set_printoptions(threshold='nan')
mat = scipy.io.loadmat('hw1data.mat')

data = mat['X'].astype(np.float64)
labels = mat['Y'].astype(int)

split = 500
total = 2000
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
    T = split*3
    w = [0]*784
    for t in range(1,T):
        if t % 1000 == 0:    
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

def perceptron_v2(digit):
    T = split*5
    w = [0]
    w.append([0]*784)
    c = [0, 0]
    k = 1
    for t in range(1,T):
        if t % 1000 == 0:
            print t
        i = (t % n + 1)
        y = -1
        if (labels[i] == digit):
            y = 1
        x = data[i]

        if (y * (np.dot(w[k],x)) <= 0):
            w.append(w[k] + y*x)
            c.append(1)
            k = k + 1
        else:
            c[k] = c[k] + 1
    return w, c, k


def kernel(degree, x1, x2):
    c = 0
    return (np.dot(x1, x2) + c) ** degree


def kernel_perceptron(digit):
    T = split*1
    a = [0] * (n+1)
    for t in range(1,T):
        if t % 100 == 0:
            print t
        i = (t % n + 1)
        x = data[i]
        y = -1
        if (labels[i] == digit):
            y = 1
        sum = 0.0
        for j in range(1, n):
            yj = -1
            if (labels[j] == digit):
                yj = 1
            sum = sum + a[j] * yj * kernel(1, data[j], x)
        yguess = -1
        if sum >= 0:
            yguess = 1
        if yguess != y:
            a[i] = a[i] + 1
    return a


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
        
def test_p2():
    w = [0]*10
    c = [0]*10
    k = [0]*10
    testSet = data[split:total]
    for digit in range(10):
        w[digit], c[digit], k[digit] = perceptron_v2(digit)


    correctcnt = 0
    wrongcnt = 0
    for i in range(len(testSet)):
        x = testSet[i]

        guess = 0
        for digit in range(10):
            sum = 0.0
            for j in range(1, k[digit]+1):
                sign = -1
                if (np.dot(w[digit][j], x) >= 0):
                    sign = 1
                sum = sum + c[digit][j] * sign
            if (sum >= 0):
                guess = digit

        if guess == labels[i + split, 0]:
            correctcnt += 1
        else:
            wrongcnt += 1

        print "guess ", guess
        print "actual ", labels[i + split, 0]

        print correctcnt, wrongcnt
        # print w

def test_kp():
    a = [0]*10
    testSet = data[split:total]
    for digit in range(10):
        a[digit] = kernel_perceptron(digit)


    correctcnt = 0
    wrongcnt = 0
    for i in range(len(testSet)):
        x = testSet[i]
        y = -1
        if (labels[i + split] == digit):
            y = 1

        guess = 0
        f = -1 * float("inf")
        for digit in range(10):
            sum = 0.0
            for j in range(1, split):
                yj = -1
                if (labels[j] == digit):
                    yj = 1
                sum = sum + a[digit][j] * yj * kernel(1, data[j], x)
            if (sum >= f):
                f = sum
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
    #test_p0()
    #test_p1()
    #test_p2()
    test_kp()

main()