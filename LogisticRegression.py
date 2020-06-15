# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 02:02:18 2020

@author: kingslayer
"""

#LOGISTIC REGRESSION

#importing the librararies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv(r"Social_Network_Ads.csv")

#creating matrices of features
X=dataset.iloc[:,[1,2,3]].values
#creating dependatnt vector
y=dataset.iloc[:,-1].values

#Feature Scaling
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()



#splittiong into train,test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)

#fitting logistic regression to train set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#predicting
y_pred=classifier.predict(X_test)

#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
