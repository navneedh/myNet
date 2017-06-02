import csv
import numpy as np
import pandas as pd
import pylab as p
import matplotlib.pyplot as plt
#from plot_learning_curve import plot_learning_curve as curve
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import subprocess

def normalize(s):
    return (s + s.mean())/s.std()

df = pd.read_csv('../trainingData/titanic.csv', header = 0)

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

survive = df.Survived
df.drop('Survived', axis = 1)
df.insert(8,'Y',survive)
