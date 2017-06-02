import numpy as np
import pandas as pd
import csv
import neuralNetwork as net
from clean import df

data=df.as_matrix()
#data = pd.read_csv('../trainingData/testData1.csv', header=0).as_matrix()
print(data)
X = (data[:,0:data.shape[1]-1])
Y = data[:,data.shape[1] - 1]


#need to create an input output framework
#vector one
#vector two
hLayer1 = net.ly.Layer(2, 'relu')
#vector three
outputLayer = net.ly.Layer(3, 'relu')

net.NeuralNetwork.constructNet(X, hLayer1, outputLayer)

#Flow of matrices is a 4x3 matrix, 5x4 matrix, 2x5 matrix, 1x2 matrix
#later try and implement batch normalization, different gradient descents, and activation functions

network = hLayer1.myNetwork

# network.toString()

network.train(X,Y, 'logistic')
