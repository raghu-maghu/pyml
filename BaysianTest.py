from sklearn import datasets

iris = datasets.load_iris()

print(iris.feature_names)
print(iris.target_names)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=109)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train,Y_train)

y_pred = nb.predict(X_test)

from sklearn import metrics

print(metrics.accuracy_score(Y_test,y_pred))
