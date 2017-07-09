import kmeans as km
import numpy as np
import tensorflow as tf

#hyperparameters
BATCH = 10000

x,y = km.trainingData(100,2,4)

#generate random y training cluster number values
y_training = [np.random.randint(1,15) for x in range(BATCH)]

#is PCA even useful for a two dimensional matrix
matrix = np.vstack((x,y)).T
U, S, V = np.linalg.svd(matrix, full_matrices=True)

#first principal component vector
x_training = U[:,0]


X = np.zeros([None,100])
Y = y_training

def trainingData():    

#create an ensemble method basically using different weights of three mechanisms depending on value of principal component

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None,100])
y_ = tf.placeholder(tf.float32, shape=[None,15])

def weights(dimensions):
    return tf.get_variable("W", shape=[dimensions[0], dimensions[1]], initializer=tf.contrib.layers.xavier_initializer())

def bias(dimension):
    return tf.get_variable("b", shape=[dimension], initializer=tf.zeros_initializer())

weight = {'W1': weights([100,60]), 'W2': weights([60,40]), 'W3': weights([40,10])}
biases = {'B1': bias(60), 'B2': bias(40), 'B3': bias(10)}

def neuralNet():
    l1 = tf.matmul(x,)

sess.run(tf.global_variables_initializer())
