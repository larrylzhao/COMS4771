import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab


np.set_printoptions(threshold='nan')
mat = scipy.io.loadmat('hw1data.mat')

data = mat['X'].astype(int)
labels = mat['Y'].astype(int)

total = len(data)
split = 7000
k = 1

trainingData = data[0:split]
trainingLabels = labels[0:split]
testSet = data[split:total]
testLabels = labels[split:total]

# calculate class priors
classPriors = [0.0]*10
for label in trainingLabels:
    classPriors[label[0]] += 1.0


# calculate conditionals for each digit
# mu is a 10 x 784 matrix holding the mean vectors of each digit
mu = [0]*10
for i in range(0, len(mu)):
    mu[i] = [0]*784

for i in range(0, len(trainingData)):
    digit = trainingLabels[i][0]
    for column in range(0, 784):
        mu[digit][column] = mu[digit][column] + trainingData[i][column]/classPriors[digit]
        # print column

# sigma is a 10 x 784 x 784 matrix holding the covar matrices of each digit
sigma = [0]*10

for i in range(0, len(trainingData)):
    digit = trainingLabels[i][0]
    xi = np.matrix(trainingData[i]).astype(float)
    mutemp = np.matrix(mu[digit]).astype(float)
    diff = (xi - mutemp)/classPriors[digit]
    diffT = diff.transpose()
    prod = diffT * diff
    sigma[digit] = sigma[digit] + prod

correctcnt = 0
wrongcnt = 0
# calculate the probabilities and return the highest
for i in range(len(testSet)):
    x = np.matrix(testSet[i]).astype(float)
    highestDigit = 0
    highestProb = 99999999.9
    for digit in range(0, 10):
        mutemp = np.matrix(mu[digit]).astype(float)
        diff = x - mutemp
        diffT = diff.transpose()
        sigtemp = sigma[digit]
        siginv = np.linalg.inv(sigtemp + np.identity(784))
        exp = -0.5 * (diff * siginv * diffT)[0,0]

        #taking log to avoid overflow
        (sign, logdet) = np.linalg.slogdet(sigtemp + np.identity(784))
        logpi = 784 * np.log(2 * np.pi)
        p = -0.5 * (logpi + (sign * logdet) + exp) * classPriors[digit] / 10000.0

        if p < highestProb:
            highestProb = p
            highestDigit = digit

        # print exp, logdet, logpi, p, highestProb, highestDigit

    if highestDigit == testLabels[i, 0]:
        correctcnt += 1
    else:
        wrongcnt += 1

    print "guess ", highestDigit
    print "actual ", testLabels[i, 0]

    print correctcnt, wrongcnt