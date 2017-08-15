import math
import numpy as np

class Computation:

    def __init__(self, network):
        self.forPropValue = 0
        self.backPropValue = 0
        self.totalError = 0
        self.errorArray = []

#activation functions
def sigmoid(x):
    return 1.0/ (1.0 + np.exp(-x))

def tanh(x):
    return np.tanh([x]) #might only be possible with array

def linear(x):
    return x

def relu(x):
    if x < 0:
        return 0
    else:
        return x

def softmax(x): #only for multiclass and final layer
    pass

#loss functions
#actual is wrong and expected is right

def logistic(actual, expected): #binary classification should be used with sigmoid
    if expected == 1:
        return -math.log(actual) #this does natural log
    elif expected == 0:
        return -math.log(1-actual)

def hinge(actual, expected): #does not need to use sigmoid function before hinge loss, directly take linear output into hinge loss function
    return math.max(0,1-expected*actual)

def crossEntropy(actual,expected): #multiclass classification should be used with softmax
    pass

def l1(actual, expected):
    return math.pow(actual-expected,2)

def l2(actual, expected):
    return math.abs(actual-expected)

#figure out what loss function to use for relu

#derivatives

def derSigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def derLogLoss(true_error, x):
    if true_error == 1:
        return -1/x #this does natural log
    elif true_error == 0:
        return 1/(1-x)

def derRelu(x):
    if x <= 0:
        return 0
    else:
        return 1
