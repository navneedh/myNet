import numpy as np
import layer as ly

class NeuralNetwork:
    def __init__(self, layerSize, inputData):
        self.numLayers = layerSize
        self.layerArray = []
        self.input = inputData
        self.inputFeatureSize = inputData.shape[1]
        self.trials = 0
        self.computation = None
        self.weightArray = []

    def forwardProp(self):
        initialData = 1

    def train(self,data):
        pass

    def toString(self):
        for layer in self.layerArray:
            print(layer.size)
            print("||")

    def constructNet(inputData, *args, initialize='xavier', classify = 'binary'): #need to figure out how I will implmenet multiclass classification


        args, numLayers = list(args), len(args)

        #initialize network
        network = NeuralNetwork(numLayers,inputData)
        inputLayer = ly.Layer(network.inputFeatureSize, 'start', True)

        if classify == "binary":
            lastLayer = ly.Layer(1, 'end', True)

        prevSize, prevLayer = inputLayer.size, inputLayer
        args.insert(0,inputLayer)
        args.append(lastLayer)

        for layer in args[1:]:
            try:
                layer.myNetwork = network
                network.layerArray.append(layer)
                prevLayer.next = layer;
                prevLayer.weights = layer.createMatrix(layer.size, prevSize, initialize)
                #layer.aMatrix = prevLayer.weights
                network.weightArray.append(prevLayer.weights)
                print(prevLayer.weights)
                prevSize, prevLayer = layer.size, layer
            except:
                print("Wrong initializations")

class Computation:

    def __init__(self, network):
        self.forPropValue = 0
        self.backPropValue = 0
        self.totalError = 0
