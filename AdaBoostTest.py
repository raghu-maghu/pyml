from sklearn import datasets
iris = datasets.load_iris()

print(iris.feature_names)
print(iris.target_names)
X=iris.data
Y=iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.3)

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=50,learning_rate=1)

ada.fit(X_train,Y_train)

y_pred = ada.predict(X_test)

from sklearn import metrics
print(metrics.accuracy_score(Y_test,y_pred))
