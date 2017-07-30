import tensorflow as tf
import pandas as pd
import kmeans as km
import math

#create kernel matrix

#hyperparameters
TRAININGSIZE = 500
BATCH = TRAININGSIZE//50
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
    return np.norm(x-y)


#generate random y training cluster number values
y_training = [np.random.randint(1,5) for x in range(TRAININGSIZE)]

def getXVector(points, dimension, clusters):
    x,y = km.trainingData(points,dimension,clusters)

    #is PCA even useful for a two dimensional matrix
    matrix = np.vstack((x,y)).T
    U, S, V = np.linalg.svd(matrix, full_matrices=True)

    # return first principal component vector
    x_training = U[:,0]

    return x_training

x_training = [getXVector(50, 2, y) for y in y_training]

y_training_onehot = [tf.one_hot([y], 4).eval()[0] for y in y_training]

print("Finished gathering training data")
