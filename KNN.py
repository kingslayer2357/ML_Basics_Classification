# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 04:13:05 2020

@author: kingslayer
"""

#KNN

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv(r"Social_Network_Ads.csv")

#creating matrix of features
X=dataset.iloc[:,[2,3]].values
#creating dependant vector
y=dataset.iloc[:,-1].values

#splitting into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Fitting KNN
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,p=2)
classifier.fit(X_train,y_train)

#Predicting
y_pred=classifier.predict(X_test)

#Confuion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
