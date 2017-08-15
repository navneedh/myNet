import numpy as np
import layer as ly
import computation as cp
import matplotlib.pyplot as plt


#constants
actDict = {'sigmoid': cp.sigmoid, 'relu': cp.relu, 'tanh' : cp.tanh}

lossDict = {'softmax':cp.softmax, 'logistic':cp.logistic}

derDict = {'derSig' : cp.derSigmoid, 'logistic' : cp.derLogLoss, 'derRelu': cp.derRelu}

class NeuralNetwork:

    def __init__(self, layerSize, inputData):
        self.numLayers = layerSize
        self.layerArray = []
        self.layerSizes = []
        self.input = inputData
        self.inputFeatureSize = inputData.shape[1]
        self.trials = 0
        self.computation = cp.Computation(self)
        self.weightArray = []
        self.derArray = []

    def train(self,X,Y,errorFunc="logistic", learning_rate = 2.3, batchSize = 200): #probably need to create another train function for multiclass
        status = "*"
        errorVal = 0 #new change
        for batchCount in range(batchSize):
            for index in range(800): #training set size
                self.layerArray[0].neurons = X[index]
                finalValue = self.forwardProp(X[index])[0]
                print(finalValue)
                errorVal = lossDict[errorFunc](finalValue, Y[index])
                self.backwardProp(errorVal, errorFunc, Y[index], learning_rate, finalValue)
                self.computation.errorArray.append(errorVal)
                print(errorVal)
            #print(str(batchCount/batchSize) * 100 + "% Complete")
        plt.plot(self.computation.errorArray)
        plt.show()

    def forwardProp(self, inputData):
        nextVal = inputData
        for i in range(len(self.layerArray) - 1): #default sigmoid activation
            nextVal = np.dot(self.layerArray[i].weights,nextVal)
            vfunc = np.vectorize(actDict['sigmoid'])
            derfunc = np.vectorize(derDict['derSig'])
            #partial derivative calculation
            for x in range(self.layerArray[i].weights.shape[0]):
                for y in range (self.layerArray[i].weights.shape[1]):
                    self.layerArray[i].partialDer[x,y] = inputData[y] * derDict['derSig'](nextVal[x])
                    self.derArray[i][x,y] = self.layerArray[i].partialDer[x,y]

            self.layerArray[i+1].preActNeurons = derfunc(nextVal)
            nextVal = vfunc(nextVal)
            inputData = nextVal
            self.layerArray[i+1].neurons = nextVal
        return nextVal

    def backwardProp(self, error, loss_function, true_error, learning_rate, finalValue):
        totalErrorDerivative = derDict[loss_function](true_error, finalValue)
        #print(totalErrorDerivative)
        prog = np.array([totalErrorDerivative])
        for index in reversed(range(len(self.derArray))):
            finalGradients = []
            for col in range(self.derArray[index].shape[1]):
                finalGradients.append(np.multiply(prog, self.derArray[index][:, col]))
            finalGradients = np.array(finalGradients).T

            #preparing the next prog matrix for next backprop step
            temp = []
            for col in range(self.weightArray[index].T.shape[1]):
                temp.append(np.multiply(self.layerArray[index].preActNeurons, self.weightArray[index].T[:,col]))

            prog = np.dot(prog, temp)

            #gradient descent
            self.weightArray[index] = self.weightArray[index] - np.multiply(finalGradients, learning_rate)
            #print(self.weightArray[index])

    def toString(self):
        for layer in self.layerArray:
            print(layer.neurons)

    def constructNet(inputData, *args, initialize='gaussian', classify = 'binary'): #need to figure out how I will implmenet multiclass classification
        args, numLayers = list(args), len(args)

        #initialize network
        network = NeuralNetwork(numLayers,inputData)
        inputLayer = ly.Layer(network.inputFeatureSize, 'start', True)

        if classify == "binary":
            lastLayer = ly.Layer(1, 'sigmoid', True)
        elif classify == "multiclass": #still in development
            lastLayer = ly.Layer(10000, 'softmax', True) #100 is a filler number

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
                prevLayer.partialDer = layer.createMatrix(layer.size, prevSize, 'zeros')
                #layer.aMatrix = prevLayer.weights
                network.derArray.append(prevLayer.partialDer)
                network.weightArray.append(prevLayer.weights)
                network.layerSizes.append(layer.size)
                prevSize, prevLayer = layer.size, layer
            except:
                print("Wrong initializations")
