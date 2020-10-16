from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
n_cluster = 10
kmeans = KMeans(n_clusters=n_cluster)
kmeans.fit(X)
y = kmeans.predict(X)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            marker='^',s=100,linewidth=2,edgecolors='k')
plt.scatter(X[:,0],X[:,1],c=y,marker='o',s=13)
plt.show()