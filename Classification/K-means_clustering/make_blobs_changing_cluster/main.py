from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
a = 300  # 극단값들이 줄어듬
b = 10  # 데이터들이 정가운데 뭉치는 정도
c = 0.1  # 가운데 놈의 뭉침 정도 높을수록 풀림.
X_varied, y_varied =\
    make_blobs(n_samples=200,cluster_std=[a,b,c],random_state=170)
plt.scatter(X_varied[:,0], X_varied[:,1], y_varied)
plt.show()
fig = plt.gcf()
fig.savefig(str(a)+"_"+str(b)+"_"+str(c)+".png",dpi=fig.dpi)


#y_pred = KMeans(n_cluster=3,random_state=0).fit_predict(X_varied)