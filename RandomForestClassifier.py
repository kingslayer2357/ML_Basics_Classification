# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 03:18:38 2020

@author: kingslayer
"""

#RANDOM FOREST

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv(r"Social_Network_Ads.csv")

#creating matrix of features and vector
X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

#splitting into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#fitting
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)

#predicting
y_pred=classifier.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
