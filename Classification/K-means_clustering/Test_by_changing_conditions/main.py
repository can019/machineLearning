from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.scatter(X[:,0], X[:,1], c=y,s=60,edgecolors='k')
plt.show()
X_varied
, y_varied = make_blobs n_samples =200 , cluster_std =[1.0, 2.5, 0.5], random_stae =
y_pred = KMeans(n_cluster=3, random_state=0).fit_predict(X_varied)

plt.scatter(X[:,0], X[:,1], c=y_pred,s=60,edgecolors='k')
kmeans - Kmeans(n_cluster=2)
kmeans.fit(X)
y_pred = kmeans.predict(X)
