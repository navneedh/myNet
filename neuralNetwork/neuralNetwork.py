import numpy as np
import layer as ly
import computation as cp
import matplotlib.pyplot as plt


#constants
actDict = {'sigmoid': cp.sigmoid, 'relu': cp.relu, 'tanh' : cp.tanh}

lossDict = {'softmax':cp.softmax, 'logistic':cp.logistic}

derDict = {'sigmoid' : cp.derSigmoid, 'logistic' : cp.derLogLoss, 'derRelu': cp.derRelu}

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
        self.biasArray = []
        self.preActArray = []

    def train(self,X,Y,errorFunc="logistic", learning_rate = 2.3, batchSize = 200): #probably need to create another train function for multiclass
        status = "*"
        error = 0 #new change
        for batchCount in range(batchSize):
            for index in range(4): #training set size
                self.layerArray[0].neurons = X[index]
                weightgradient, biasgradient, result = self.propagate(X[index],Y[index])
                print(result)
                self.optimize(weightgradient, biasgradient)
                error = lossDict['logistic'](result[0], Y[index])
                self.computation.errorArray.append(error)
                print("Error:", error)
            #print(str(batchCount/batchSize) * 100 + "% Complete")
        plt.plot(self.computation.errorArray)
        plt.show()

    def propagate(self, x, y):
        nextVal = x
        for i in range(len(self.layerArray) - 1): #default sigmoid activation
            nextVal = np.dot(self.layerArray[i].weights,nextVal) + self.biasArray[i]
            self.preActArray.append(nextVal)
            vfunc = np.vectorize(actDict['sigmoid'])
            nextVal = vfunc(nextVal)
            #inputData = nextVal
            self.layerArray[i+1].neurons = nextVal

        result = nextVal
        biasgradient = []
        weightgradient = []
        lastLayerGradient = derDict['logistic'](y, nextVal) * derDict['sigmoid'](self.preActArray[-1])
        biasGrad = lastLayerGradient
        weightGrad = np.dot(lastLayerGradient, self.layerArray[-2].neurons.reshape(1,self.layerArray[-2].neurons.shape[0])) #need to change from constant 1 for multiclass
        biasgradient.append(biasGrad)
        weightgradient.append(weightGrad)
        prog = np.array([lastLayerGradient])
        for index in reversed(range(len(self.weightArray))):
            x = np.dot(self.weightArray[index].T, biasGrad)
            biasGrad = np.dot(self.weightArray[index].T, biasGrad) * derDict['sigmoid'](self.preActArray[index - 1])
            l = self.layerArray[index - 1].neurons
            weightGrad = np.outer(biasGrad, l)
            biasgradient.append(biasGrad)
            weightgradient.append(weightGrad)

        return list(reversed(weightgradient[:-1])), list(reversed(biasgradient[:-1])), result

    def optimize(self, weightgradient, biasgradient, learning_rate=5): #need to fix this method
        #gradient descent


        for index in (range(len(self.weightArray))):
            self.weightArray[index] = self.weightArray[index - len(self.weightArray)] - np.multiply(weightgradient[index], learning_rate)
            self.biasArray[index] = self.biasArray[index] - np.multiply(biasgradient[index], learning_rate)

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
                prevLayer.bias = np.zeros(layer.size) #should not be initialized at zero
                #layer.aMatrix = prevLayer.weights
                network.biasArray.append(prevLayer.bias)
                network.weightArray.append(prevLayer.weights)
                network.layerSizes.append(layer.size)
                prevSize, prevLayer = layer.size, layer
            except:
                print("Wrong initializations")
