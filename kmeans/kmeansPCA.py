import kmeans as km
import numpy as np
import tensorflow as tf

#hyperparameters
BATCH = 10000

x,y = km.trainingData(100,2,4)

#generate random y training cluster number values
y_traning = [np.random.randint(1,15) for x in range(BATCH)]

#is PCA even useful for a two dimensional matrix
matrix = np.vstack((x,y)).T
U, S, V = np.linalg.svd(matrix, full_matrices=True)

#first principal component vector
x_training = U[:,0]

#create an ensemble method basically using different weights of three mechsnisms depending on value of principal component

def neuralNet():
