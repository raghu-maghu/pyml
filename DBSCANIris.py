from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.cluster import DBSCAN
#xdbscan = DBSCAN(random_state=111)

dbscan=DBSCAN(eps=0.5, metric='euclidean', min_samples=5)

dbscan.fit(iris.data)


from sklearn.decomposition import PCA
import matplotlib.pyplot as pl
pca = PCA(n_components=2).fit(iris.data)
pca_2d = pca.transform(iris.data)
for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0:
        c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r', marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')
pl.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Noise'])
pl.title('DBSCAN finds 2 clusters and noise')
pl.show()