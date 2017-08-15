import numpy as np
import pandas as pd
import csv
import neuralNetwork as net
#from clean import df


df = pd.read_csv('../trainingData/xor.csv', header=0).as_matrix()


#vector one
#vector two
hLayer1 = net.ly.Layer(2, 'sigmoid')
hLayer3 = net.ly.Layer(4, 'sigmoid')
#vector three
outputLayer = net.ly.Layer(3, 'sigmoid')


net.NeuralNetwork.constructNet(X, hLayer1, hLayer3, outputLayer)

#Flow of matrices is a 4x3 matrix, 5x4 matrix, 2x5 matrix, 1x2 matrix
#later try and implement batch normalization, different gradient descents, and activation functions

network = hLayer1.myNetwork

# network.toString()

network.train(X,Y, 'logistic')
