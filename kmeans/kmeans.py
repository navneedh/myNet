import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import math

#hyperparameters
NUM_CENTROIDS = 4
ITERATIONS = 1000

#helper functions
def calcDistance(arr1, arr2):
    return math.sqrt(math.pow(arr1[0] - arr2[0],2) + math.pow(arr1[1] - arr2[1],2))

def average(vals):
    return np.mean(vals, axis=0)

def trainingData(values, dimensions, blobNum):
    data = (make_blobs(values,dimensions,blobNum)[0])
    x, y = zip(*data)
    return x, y

def train(x,y,NUM_CENTROIDS=4, ITERATIONS=1000):
    global centroids
    global seedCentroids
    data = []
    for i in range(len(x)):
        data.append([[x[i],y[i]],0])

    #create bounding box
    minPair = [min(x), min(y)]
    maxPair = [max(x),max(y)]

    centroids = []
    #initialize random centroids
    for i in range(NUM_CENTROIDS):
        centroids.append([np.random.randint(minPair[0],maxPair[0]),np.random.randint(minPair[1],maxPair[1])])


    #store seed centroids
    seedCentroids = list(centroids)

    for _ in range(ITERATIONS):
        valsToAvg = {(x[0]):[] for x in enumerate(centroids)}
        for i1, elem in enumerate(data):
            distance = 10000
            for i2, c in enumerate(centroids):
                val = calcDistance(c,elem[0])
                if val < distance:
                    distance = val
                    data[i1][1] = i2
            valsToAvg[(data[i1][1])].append(elem[0])

        centroids_copy = []
        for key in valsToAvg.keys():
            if len(valsToAvg[key]) == 0:
                centroids_copy.append(centroids[key])
            else:
                # print(average(valsToAvg[key]).tolist())
                centroids_copy.append(average(valsToAvg[key]).tolist())

        centroids = centroids_copy


def trainFull(values=100, dimensions=2, blobNum=4, NUM_CENTROIDS=4, ITERATIONS=1000):
    global x
    global y
    global centroids
    global seedCentroids
    data = (make_blobs(values,dimensions,blobNum)[0])
    x, y = zip(*data)
    train(x,y,NUM_CENTROIDS,ITERATIONS)

def graphFull(x,y):
    plt.plot(x,y, 'ro')
    a, b = zip(*centroids)
    plt.plot(a, b, 'g^', linewidth=2.0)
    plt.show()

def graph(x,y):
    plt.plot(x,y, 'ro')
    plt.show()

if __name__ == '__main__':
    train()
    graph()
    print("Initial:", seedCentroids)
    print("Final:", centroids)
