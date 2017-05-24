import numpy as np
import pandas as pd
import csv
import neuralNetwork as net

data = pd.read_csv('testData1.csv', header=0)

#need to create an input output framework
#vector one
inputLayer = net.ly.Layer(4, 'linear')
#vector two
hLayer1 = net.ly.Layer(5, 'linear')
#vector three
outputLayer = net.ly.Layer(2, 'linear')

net.NeuralNetwork.constructNet(data, inputLayer, hLayer1, outputLayer)

#Flow of matrices is a 4x3 matrix, 5x4 matrix, 2x5 matrix, 1x2 matrix
#later try and implement batch normalization, different gradient descents, and activation functions

network = hLayer1.myNetwork
network.train(data)
