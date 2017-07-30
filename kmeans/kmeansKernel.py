import tensorflow as tf
import pandas as pd
import kmeans as km
import numpy as np
import math

#create kernel matrix

#hyperparameters
TRAININGSIZE = 500
BATCH = 20
EPOCHS = 1000
display_step = 1

#helper kernel functions
def polykernel(x,y):
    c = 2
    d = 3
    return math.pow(np.dot(x,y) + c, d)

def rbfkernel(x,y):
    return

def kernel1(x,y):
    return np.linalg.norm(x-y)

def genkernel(vec, veclist, kernel = kernel1):
    return [kernel(vec,np.array(vec1)) for vec1 in veclist]

#generate random y training cluster number values
y_training = [np.random.randint(1,5) for x in range(TRAININGSIZE)]

def getXVector(points, dimension, clusters):
    x,y = km.trainingData(points,dimension,clusters)
    xy = list(zip(x,y))
    print(xy)
    results = np.array([np.array(genkernel(np.array(xy[i]), xy)) for i in range(len(xy))])
    print(results.shape)

    U, S, V = np.linalg.svd(results, full_matrices=True)
    print(S)
    # return first principal component vector
    x_training = U[:,0]

    return x_training

x_training = [getXVector(50, 2, y) for y in y_training]

y_training_onehot = [tf.one_hot([y], 4).eval()[0] for y in y_training]

print("Finished gathering training data")
