import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('C:/Users/rs6134/Downloads/MLDATA/iris.csv')

x = df.iloc[:,[0,1,2,3]].values

kmeans5 = KMeans(n_clusters=5)

y_pred = kmeans5.fit_predict(x)

print(y_pred)

print(kmeans5.cluster_centers_)
Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()