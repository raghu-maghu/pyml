import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

#df=pd.read_csv("C:/Users/rs6134/Downloads/MLDATA/iris.csv",names=['sepal length','sepal width', 'petal length','petal width','target'])

df=pd.read_csv(url,names=['sepal length','sepal width', 'petal length','petal width','target'])
print(df)

from sklearn.preprocessing import StandardScaler
features = ['sepal length','sepal width', 'petal length','petal width']
x = df.loc[:,features].values
print("Print x \n",x)

x = StandardScaler().fit_transform(x)
y = df.loc[:,'target'].values

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(x)

final_comp = pd.DataFrame(data=pca_components,columns=["PCA1","PCA2"])
print("Final Comp \n",final_comp.head(3))

print(pca.explained_variance_ratio_)