import math

class Computation:

    def __init__(self, network):
        self.forPropValue = 0
        self.backPropValue = 0
        self.totalError = 0

#activation functions
def sigmoid(x):
    return 1/ (1 + math.exp(-x))

def tanh(x):
    #need to fill this in
    return None

def linear(x):
    return x

def relu(x):
    if x < 0:
        return 0
    else:
        return x

def softmax(x):
    pass

#loss functions

def logistic(actual, expected): #binary classification should be used with sigmoid
    if actual == 1:
        return -math.log(expected)
    elif actual == 0:
        return -math.log(1-expected)

def hinge(actual, expected): #does not need to use sigmoid function before hinge loss, directly take linear output into hinge loss function
    return math.max(0,1-expected*actual)ß

def crossEntropy(actual,expected): #multiclass classification should be used with softmax
    pass

def l1(actual, expected):
    return math.pow(actual-expected,2)

def l2(actual, expected):
    return math.abs(actual-expected)

#figure out what loss function to use for relu
