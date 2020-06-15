# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 00:49:17 2020

@author: kingslayer
"""

#SVM
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv(r"Social_Network_Ads.csv")

X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

#Splitting into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Fitting SVM
from sklearn.svm import SVC
classifier=SVC(kernel="rbf",random_state=0)
classifier.fit(X_train,y_train)

#predicting
y_pred=classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
