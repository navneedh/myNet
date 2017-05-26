import numpy as np
import layer as ly
import math

class NeuralNetwork:
    def __init__(self, layerSize, inputData):
        self.numLayers = layerSize
        self.layerArray = []
        self.layerSizes = []
        self.input = inputData
        self.inputFeatureSize = inputData.shape[1]
        self.trials = 0
        self.computation = Computation(self)
        self.weightArray = []

    def train(self,data,error="logistic"): #probably need to create another train function for multiclass
        self.layerArray[0].neurons = data[0]
        self.forwardProp(data[0], layerArray[0].actFunc)
        if error == "logistic":
            print("test")

        #for sample in data:
            #forwardProp()
            #backwardProp()

    def forwardProp(self, inputData, activation):
        oneVal = inputData
        for i in range(len(self.layerArray) - 1): #default sigmoid activation
            oneVal = np.dot(self.layerArray[i].weights,oneVal)
            vfunc = np.vectorize(actDict[])
            oneVal = vfunc(oneVal)
            #oneVal = np.apply_along_axis(self.computation.relu, 0, [oneVal])[0]
            self.layerArray[i+1].neurons = oneVal
            print(oneVal)
            print("**********")
        return oneVal


    def toString(self):
        for layer in self.layerArray:
            print(layer.neurons)

    def constructNet(inputData, *args, initialize='gaussian', classify = 'binary'): #need to figure out how I will implmenet multiclass classification
        args, numLayers = list(args), len(args)

        #initialize network
        network = NeuralNetwork(numLayers,inputData)
        inputLayer = ly.Layer(network.inputFeatureSize, 'start', True)

        if classify == "binary":
            lastLayer = ly.Layer(1, 'end', True)
        elif classify == "multiclass":
            lastLayer = ly.Layer(10000, 'end', True) #100 is a filler number

        prevSize, prevLayer = inputLayer.size, inputLayer
        args.insert(0,inputLayer)
        args.append(lastLayer)
        network.layerArray.insert(0,inputLayer)
        network.layerSizes.append(inputLayer.size)

        for layer in args[1:]:
            try:
                layer.myNetwork = network
                network.layerArray.append(layer)
                prevLayer.next = layer;
                prevLayer.weights = layer.createMatrix(layer.size, prevSize, initialize)
                #layer.aMatrix = prevLayer.weights
                network.weightArray.append(prevLayer.weights)
                network.layerSizes.append(layer.size)
                prevSize, prevLayer = layer.size, layer
            except:
                print("Wrong initializations")



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
