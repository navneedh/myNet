import numpy as np

numToLetter = {}
numbers = ['1','2','3','4','5','6','7','8','9','0',',','?','/','\'']

for i in range(0,26):
    numToLetter[chr(97 + i)] = i

def cosDistance(vec1, vec2):
    return np.inner(vec1,vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))

def cleanString(string):
    string = "".join([char for char in string if char not in numbers])
    return (string.replace(" ", "").strip().lower().replace("/", "").replace("?","").replace("&","").replace("(","").replace(")", "").replace(".","").replace("-",""))

def word2vec(cleanString):
    vector = np.zeros((26,))
    for index,letter in enumerate(cleanString):
        vector[numToLetter[letter]] += 1
    return vector

word1 = 'this is a test'
word2 = 'this i test'

word1 = word2vec(cleanString(word1))
word2 = word2vec(cleanString(word2))

#Create Weight matrix for a single image
def create_W(x):
    w = np.outer(x,x)
    np.fill_diagonal(w,0)
    return w

#Update
def update(w,y_vec,theta=0.5):
    u = np.dot(w,y_vec) - theta
    u[u<1] = 0
    #try using a relu instead to preserve using numbers other than 1 and -1
    return u


#The following is training pipeline
#Initial setting
def hopfield(weights, testWord,theta=0.5, time=1000):
    for _ in range(time):
        y_vec = update(w=weights,y_vec=testWord,theta=theta)
    return y_vec

#Hopfield network starts!
update = hopfield(create_W(word1), word2, theta=0.5,time=20000)

print("Actual: ", update)
print("Expected: ", word1)
print("Final Error: ", cosDistance(np.array(update), np.array(word1)))
