import pandas as pd
from sklearn import datasets

df = datasets.load_iris()

feature_names = df.feature_names
target_names= df.target_names

print ("Feature Names \n", feature_names)
print ("Target Names \n", target_names)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df.data, df.target,train_size=0.3,random_state=109)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train,Y_train)

y_pred = gnb.predict(X_test)

from sklearn import metrics
print ("Accuracy", metrics.accuracy_score(Y_test,y_pred))
