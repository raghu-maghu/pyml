# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:40:12 2021

@author: Dell
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

# load dataset
pima = pd.read_csv("C:\\Users\\rs6134\\Downloads\\MLDATA\\pima-indians-diabetes.csv", header=None, names=col_names)

print(pima.head())

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']

X = pima[feature_cols] # Features
y = pima.label # Target variable



X_train,X_test,y_train,y_test =train_test_split(X ,y ,test_size=0.20,random_state=0)


# instantiate the model (using the default parameters)
model = LogisticRegression()

# fit the model with data
model.fit(X_train,y_train)  #find the best/optimum parameters from the data

y_pred=model.predict(X_test)
print(y_pred)
# confusion matrix
# print the accuracy
cm=confusion_matrix(y_test,y_pred)
print('Confusion Matrix:\n',cm)
TP=cm[0][0]
FP=cm[0][1]
FN=cm[1][0]
TN=cm[1][1]
print('False Positive \n {}'.format(FP))
print('False Negative \n {}'.format(FN))
print('True Positive \n {}'.format(TP))
print('True Negative \n {}'.format(TN))
TPR=TP/(TP+FN)
print('Senstivity \n {}'.format(TPR))
TNR=TN/(TN+FP)
print('Specificity \n {}'.format(TNR))
Precision=TP/(TP+FP)
print('Precision \n {}'.format(Precision))
Recall=TP/(TP+FN)
print('Recall \n {}'.format(Recall))
Acc=(TP+TN)/(TP+TN+FP+FN)
print('Accuracy \n {}'.format(Acc))


sns.pairplot(pima, hue= 'label')