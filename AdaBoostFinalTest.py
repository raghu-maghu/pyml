from sklearn import datasets
iris = datasets.load_iris()

print(iris.feature_names)
print(iris.target_names)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, train_size=0.2)

from sklearn.ensemble import AdaBoostClassifier

adr = AdaBoostClassifier(n_estimators=50,learning_rate=1)

adr.fit(X_train,Y_train)

y_pred = adr.predict(X_test)

from sklearn import metrics
print ("Metrics ===\n", metrics.accuracy_score(Y_test,y_pred))