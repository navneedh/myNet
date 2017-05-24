import numpy as np
import math
import neuralNetwork as net

class Layer:
    def __init__(self, size, activationFunc, iLayer = False):
        self.input = iLayer
        self.size = size
        self.neurons = np.zeros(size);
        self.weights = None
        self.actFunc = activationFunc
        self.previous = None
        self.myNetwork = None
        self.next = None

    def createLayer(size, actFunc):
        return Layer(size, actFunc)

    def sigmoid(x):
        return 1/ (1 + math.exp(-x))

    def tanh(x):
        return None

    def linear(x):
        return x

    def relu(x):
        if x < 0:
            return 0
        else:
            return x

    def createMatrix(self, x, y, initialize):
        if initialize == 'xavier':
            return [x,y]
        else:
            print("Other initializations do not work at this time")

    activationFunctions = {'linear': linear, 'sigmoid': sigmoid, 'tanh':tanh}
