from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
a = 1.0  # 극단값들이 줄어듬
b = 2.5  # 데이터들이 정가운데 뭉치는 정도
c = 0.5  # 가운데 놈의 뭉침 정도 높을수록 풀림.
cluster = 3
X_varied, y_varied =\
    make_blobs(n_samples=200,cluster_std=[a,b,c],random_state=170)
y_pred = KMeans(n_clusters=cluster,random_state=0).fit_predict(X_varied)

plt.scatter(X_varied[:,0], X_varied[:,1], c=y_pred)
plt.show()
#fig = plt.gcf()
#fig.savefig(str(a)+"_"+str(b)+"_"+str(c)+"_cluster_"+str(cluster)+
            #".png", dpi=fig.dpi)


