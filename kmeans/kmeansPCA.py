import kmeans as km
import numpy as np
import tensorflow as tf

#hyperparameters
BATCH = 100

#generate random y training cluster number values
y_training = [np.random.randint(1,15) for x in range(BATCH)]

def getXVector(points, dimension, clusters):
    x,y = km.trainingData(points,dimension,clusters)

    #is PCA even useful for a two dimensional matrix
    matrix = np.vstack((x,y)).T
    U, S, V = np.linalg.svd(matrix, full_matrices=True)

    #first principal component vector
    x_training = U[:,0]
    return x_training

x_training = [getXVector(100, 2, y) for y in y_training]

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
    x = tf.nn.dropout(x,20)
    l1 = tf.relu(tf.matmul(x,weights['W1']) + biases['B1'])

    l1 = tf.nn.dropout(l1,20)
    l2 = tf.relu(tf.matmul(l1,weights['W2']) + biases['B2'])

    l2 = tf.nn.droupout(l2,20)
    l3 = tf.matmul(l2, weights['W3']) + biases['B3']

    return l3

result = neuralNet()

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)


sess.run(tf.global_variables_initializer())
