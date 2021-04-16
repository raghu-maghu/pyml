import pandas as pd

df = pd.read_csv("C:/Users/rs6134/Downloads/MLDATA/petrol_consumption.csv")

print(df.head)

X=df.iloc[:, 0:4].values
Y=df.iloc[:,4].values

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.ensemble import RandomForestRegressor

rd = RandomForestRegressor(n_estimators=20,random_state=0)

rd.fit(X_train,Y_train)

y_pred = rd.predict(X_test)

from sklearn import metrics
print(metrics.mean_squared_error(Y_test,y_pred))