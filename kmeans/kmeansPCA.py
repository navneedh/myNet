import kmeans as km
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#run a standard data set through this neural network to see if learning happens
#get summary of data to tensorboard
#cache training and test data to csv files

tf.InteractiveSession()

#hyperparameters
TRAININGSIZE = 1000
BATCH = 20
EPOCHS = 1000
display_step = 1

#create an ensemble method basically using different weights of three mechanisms depending on value of principal component

#generate random y training cluster number values
y_training = [np.random.randint(0,4) for x in range(TRAININGSIZE)]

def getXVector(points, dimension, clusters):
    x,y = km.trainingData(points,dimension,clusters)

    #is PCA even useful for a two dimensional matrix
    matrix = np.vstack((x,y)).T
    U, S, V = np.linalg.svd(matrix, full_matrices=True)

    # return first principal component vector
    x_training = U[:,0]

    return x_training

x_training = [getXVector(100, 2, y+1) for y in y_training]

y_training_onehot = [tf.one_hot([y], 4).eval()[0] for y in y_training]

print("Finished gathering training data")

x = tf.placeholder(tf.float32, shape=[None,100])
y_ = tf.placeholder(tf.float32, shape=[None,4])

def weights(dimensions):
    return tf.Variable(tf.random_normal([dimensions[0], dimensions[1]],stddev=0.5))

def bias(dimension):
    return tf.Variable(tf.random_normal([dimension], stddev=0.5))

weights = {'W1':weights([100,50]),'W2':weights([50,35]), 'W3': weights([35,20]), 'W4': weights([20,4])}
biases = {'B1': bias(50), 'B2': bias(35), 'B3': bias(20),  'B4': bias(4)}

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
        testX = getXVector(100,2,number).T
        testX = np.reshape(testX, (1,100))
        prediction = (sess.run(correct_prediction, feed_dict={x: testX}))
        print("Prediction:", prediction)
        print("Correct:", number)
        if number == prediction:
            totalCorrect += 1
            print("It works")

    print("Testing Accuracy:", totalCorrect/1000)

    plt.show()
