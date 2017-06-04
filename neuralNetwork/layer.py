import numpy as np
import math
import neuralNetwork as net

class Layer:
    def __init__(self, size, activationFunc, iLayer = False):
        self.input = iLayer
        self.size = size
        self.neurons = np.zeros(size)
        self.weights = None
        self.actFunc = activationFunc
        self.myNetwork = None
        self.partialDer = None
        self.previous = None
        self.next = None
        self.preActNeurons = np.zeros(size)

    def createLayer(size, actFunc):
        return Layer(size, actFunc)

    def createMatrix(self, x, y, initialize):
        if initialize == 'gaussian':
            initialArray = np.random.normal(0,0.8,x*y)
            matrix = initialArray.reshape((x,y))
            return matrix
        elif initialize == 'zeros':
            initialArray = np.zeros(x*y)
            matrix = initialArray.reshape((x,y))
            return matrix
        else:
            print("Other initializations do not work at this time")
