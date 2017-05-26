import math

class Computation:

    def __init__(self, network):
        self.forPropValue = 0
        self.backPropValue = 0
        self.totalError = 0

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
