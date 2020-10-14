from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.scatter(X[:,0], X[:,1], c=y,s=60,edgecolors='k')
plt.show()
"""
plt.scatter(X[:,0], X[:,1], c=y_pred,s=60,edgecolors='k')
kmeans - Kmeans(n_cluster=2)
kmeans.fit(X)
y_pred = kmeans.predict(X)
"""