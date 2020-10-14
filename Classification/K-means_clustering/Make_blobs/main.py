from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

plt.title("5 Cluster Data")
X,y = make_blobs(n_samples=500, n_features=2, centers=5, random_state=1)
plt.scatter(X[:,0],X[:,1], marker='o',c='y',s=100,
            edgecolors="k", linewidth=2)
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
plt.show()