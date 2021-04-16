import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
dataset = pd.read_csv(url,names=['sepal length','sepal width', 'petal length','petal width','target'])
print(dataset.head)

from sklearn.preprocessing import StandardScaler
features = ['sepal length','sepal width', 'petal length','petal width']
x = dataset.loc[:,features].values
print("Print x \n",x)

y = dataset.loc[:,'target'].values
print("Print y \n",y)

x = StandardScaler().fit_transform(x)
print('Standard Features\n',x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDF = pd.DataFrame(data=principalComponents, columns=['PCA1', 'PCA2'])

finalDF = pd.concat([principalDF,dataset[['target']]], axis=1)

print(pca.explained_variance_ratio_)