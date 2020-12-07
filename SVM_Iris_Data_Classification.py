#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:18:18 2020

@author: maan
"""

# Import libaries and datasets
import pandas as pd 
from sklearn.datasets import load_iris
iris = load_iris()

# Analysis of data and adding columns
df = pd.DataFrame(iris.data, columns = iris.feature_names)
b = df['target'] = iris.target
iris.target_names
a = df[df.target==2].head()
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

# Plotting Graphs
import matplotlib.pyplot as plt

# Classifing data basis on target_names 
df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

# Plotting Graph Between sepal lenth v/s sepal width
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color = 'green', marker = '+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color = 'blue', marker = '*')
plt.show()

# Plotting Graph Between petal length v/s petal width 
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color = 'green', marker = '+')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color = 'blue', marker = '*')
plt.show()

# Spliting the datasets
from sklearn.model_selection import train_test_split

# Droping some columns
X = df.drop(['target', 'flower_name'],axis = 'columns')
Y = df.target

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2)

# Importing SVM
from sklearn.svm import SVC
model = SVC()

model.fit(X_train,Y_train)

# Finding accuracy of test datasets
S = model.score(X_test,Y_test)