import kmeans as km
import numpy as np
import tensorflow as tf
import pandas as pd

#run a standard data set through this neural network to see if learning happens
#get summary of data to tensorboard
#cache training and test data to csv files

def normalize(s):
    return (s + s.mean())/s.std()

df = pd.read_csv('../../mlPlay/titanic/train.csv', header = 0)

df.Embarked = df.Embarked.fillna("S")
df.loc[df.Embarked == "S", "Embarked"] = 0
df.loc[df.Embarked == "C", "Embarked"] = 1
df.loc[df.Embarked == "Q", "Embarked"] = 2
df.loc[df.Sex == "male", "Sex"] = 1
df.loc[df.Sex == "female", "Sex"] = 0

df.Age = df.Age.fillna(df['Age'].median())

df.Age = normalize(df.Age)
df.Sex = normalize(df.Sex)
df.Embarked = normalize(df.Embarked)
df.Pclass = normalize(df.Pclass)
df.SibSp = normalize(df.SibSp)
df.Fare = normalize(df.Fare)
df = df.drop('Name',1)
df = df.drop('Cabin',1)
df = df.drop('Ticket',1)
df = df.drop('PassengerId', 1)

x_training = df.ix[:,1:]
y_training = df.ix[:,0]
#
#hyperparameters
TRAININGSIZE = 100
BATCH = TRAININGSIZE//40
EPOCHS = 100
display_step = 1
#
# #create an ensemble method basically using different weights of three mechanisms depending on value of principal component
sess = tf.InteractiveSession()
#
# #generate random y training cluster number values
# y_training = [np.random.randint(1,15) for x in range(TRAININGSIZE)]
#
# def getXVector(points, dimension, clusters):
#     x,y = km.trainingData(points,dimension,clusters)
#
#     #is PCA even useful for a two dimensional matrix
#     matrix = np.vstack((x,y)).T
#     U, S, V = np.linalg.svd(matrix, full_matrices=True)
#
#     # return first principal component vector
#     x_training = U[:,0]
#
#     return x_training
#
# x_training = [getXVector(100, 2, y) for y in y_training]
#
# #one hot encode y training vector
#
# y_training_onehot = [tf.one_hot([y], 15).eval()[0] for y in y_training]
#
# print("Finished gathering training data")

x = tf.placeholder(tf.float32, shape=[None,7])
y_ = tf.placeholder(tf.float32, shape=[None,])

def weights(dimensions):
    return tf.Variable(tf.random_normal([dimensions[0], dimensions[1]],stddev=0.5))

def bias(dimension):
    return tf.Variable(tf.random_normal([dimension], stddev=0.5))

weights = {'W1':weights([7,4]), 'W2':weights([4,3]), 'W3': weights([3,1])}
biases = {'B1': bias(4), 'B2': bias(3), 'B3': bias(1)}

init = tf.global_variables_initializer()

sess.run(init)

def neuralNet():
    #x_d = tf.nn.dropout(x,0.8) #might need to fix these hyperparameters
    l1 = tf.nn.relu(tf.matmul(x,weights['W1']) + biases['B1'])

    #l1 = tf.nn.dropout(l1,0.8)
    l2 = tf.nn.relu(tf.matmul(l1,weights['W2']) + biases['B2'])

    #use batch normalization
    #l2 = tf.nn.dropout(l2,0.8)
    return tf.matmul(l2, weights['W3']) + biases['B3']

result = neuralNet()

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=y_))
optimizer = tf.train.AdamOptimizer(0.001, 0.9).minimize(cost)
correct_prediction = tf.argmax(result,1)

avg_cost = 0
for epoch in range(EPOCHS):
    for i in range(BATCH):
        print(i)
        batch_x = ((x_training.ix[i*10:(i+4)*10, :])).as_matrix()
        batch_y = (y_training.ix[i*10:(i+4)*10]).as_matrix()
        print(batch_x)
        print(batch_y)
        # Run optimization op (backprop) and cost op (to get loss value)
        c = sess.run(cost, feed_dict={x: batch_x, y_: batch_y})
        # Compute average loss
        avg_cost += c / BATCH
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
print("Training Complete") 

#
# print("Execute Test")
# totalCorrect = 0
# for _ in range(100):
#     number = np.random.randint(2,15)
#     testX = getXVector(100,2,number).T
#     testX = np.reshape(testX, (1,100))
#     prediction = (sess.run(correct_prediction, feed_dict={x: testX}))
#     print("Correct:", number)
#     if number == prediction:
#         totalCorrect += 1
#
# print("Testing Accuracy:", totalCorrect/100)
