import tensorflow as tf
import pandas as pd
import kmeans as km
import numpy as np
import math
import matplotlib.pyplot as plt

#create kernel matrix

#hyperparameters
TRAININGSIZE = 1000
BATCH = 20
EPOCHS = 10000
display_step = 1


tf.InteractiveSession()


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
    results = np.array([np.array(genkernel(np.array(xy[i]), xy)) for i in range(len(xy))])
    results = np.mean(results,axis=0)
    #U, S, V = np.linalg.svd(results, full_matrices=True)
    # return first principal component vector
    #x_training = np.concatenate([U[:,0], U[:,1]])
    return results

x_training = [getXVector(50, 2, y) for y in y_training]

y_training_onehot = [tf.one_hot([y], 4).eval()[0] for y in y_training]
print(y_training_onehot)
print("Finished gathering training data")

x = tf.placeholder(tf.float32, shape=[None,50])
y_ = tf.placeholder(tf.float32, shape=[None,4])

def weights(dimensions):
    return tf.Variable(tf.random_normal([dimensions[0], dimensions[1]],stddev=0.5))

def bias(dimension):
    return tf.Variable(tf.random_normal([dimension], stddev=0.5))

weights = {'W1':weights([50,45]),'W2':weights([45,30]), 'W3': weights([30,15]), 'W4': weights([15,4])}
biases = {'B1': bias(45), 'B2': bias(30), 'B3': bias(15),  'B4': bias(4)}

def neuralNet():
    #x_d = tf.nn.dropout(x,0.8) #might need to fix these hyperparameters
    l1 = tf.nn.relu(tf.matmul(x,weights['W1']) + biases['B1'])

    l1 = tf.nn.dropout(l1,0.8)
    l2 = tf.nn.relu(tf.matmul(l1,weights['W2']) + biases['B2'])

    #use batch normalization
    l2 = tf.nn.dropout(l2,0.5)
    l3 = tf.nn.relu(tf.matmul(l2, weights['W3']) + biases['B3'])

    return tf.matmul(l3, weights['W4']) + biases['B4']

result = neuralNet()
cLog = []

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=y_))
optimizer = tf.train.AdamOptimizer(0.01, 0.9).minimize(cost)
correct_prediction = tf.argmax(result,1) + 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    avg_cost = 0
    for epoch in range(EPOCHS):
        seed = np.random.randint(1,TRAININGSIZE-BATCH)
        batch_x = np.array((x_training[seed:seed+BATCH]))
        batch_y = np.array(y_training_onehot[seed:seed+BATCH])
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y_: batch_y})
        # Compute average loss
        avg_cost += c / BATCH
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
            cLog.append(c)
    print("Training Complete")
    plt.plot(cLog)

    print("Execute Test")
    totalCorrect = 0
    for _ in range(1000):
        number = np.random.randint(1,5)
        testX = getXVector(50,2,number).T
        testX = np.reshape(testX, (1,50))
        prediction = (sess.run(correct_prediction, feed_dict={x: testX}))
        print("Prediction:", prediction)
        print("Correct:", number)
        if number == prediction:
            totalCorrect += 1
            print("It works")

    print("Testing Accuracy:", totalCorrect/1000)

    plt.show()
