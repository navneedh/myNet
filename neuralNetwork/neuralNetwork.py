import numpy as np
import layer as ly
import computation as cp


#constants
actDict = {'sigmoid': cp.sigmoid, 'relu': cp.relu, 'tanh' : cp.tanh}

lossDict = {'softmax':cp.softmax, 'logistic':cp.logistic}

derDict = {'derSig' : cp.derSigmoid, 'logistic' : cp.derLogLoss}

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

    def train(self,X,Y,errorFunc="logistic"): #probably need to create another train function for multiclass
        self.layerArray[0].neurons = X[0]
        finalValue = self.forwardProp(X[0])[0]
        #print(finalValue)
        #print(Y[0])
        errorVal = lossDict[errorFunc](finalValue, Y[0])
        self.backwardProp(errorVal, errorFunc)

        #for sample in data:
            #forwardProp()
            #backwardProp()

    def forwardProp(self, inputData):
        oneVal = inputData
        for i in range(len(self.layerArray) - 1): #default sigmoid activation
            oneVal = np.dot(self.layerArray[i].weights,oneVal)
            vfunc = np.vectorize(actDict[self.layerArray[i + 1].actFunc])
            print(self.layerArray[i].weights.shape)
            print(self.layerArray[i].partialDer.shape)
            print(self.layerArray[i].weights)

            #partial derivative calculation
            for x in range(self.layerArray[i].weights.shape[0]):
                for y in range (self.layerArray[i].weights.shape[1]):
                    print(x, y)
                    print(inputData)
                    print(oneVal)
                    self.layerArray[i].partialDer[x,y] = inputData[y] * derDict['derSig'](oneVal[x])

            inputData = oneVal
            oneVal = vfunc(oneVal)
            #oneVal = np.apply_along_axis(self.computation.relu, 0, [oneVal])[0]

            self.layerArray[i+1].neurons = oneVal
            # print(oneVal)
            # print("**********")
        return oneVal

    def backwardProp(self, error, loss_function):
        totalErrorDerivative = derDict[loss_function](error) #need to write the actual method


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
        elif classify == "multiclass":
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
