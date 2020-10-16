from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
a = 1.0 # 극단값이 우후죽순 늘어남.
b = 2.5  # 중앙에 모이려는 성질이 크다. 극단값들의 평균이 중앙에 위치
c =0.5 # 극단값들을 제외한 나머지 값들이 가운데로 모임
cluster = 100 # 6이상부터 억지
X_varied, y_varied =\
    make_blobs(n_samples=200,cluster_std=[a,b,c],random_state=170)
y_pred = KMeans(n_clusters=cluster,random_state=0).fit_predict(X_varied)
# plt.scatter(X_varied[:,0],X_varied[:,1],y_varied)
plt.scatter(X_varied[:,0], X_varied[:,1], c=y_pred)
plt.show()
#fig = plt.gcf()
#fig.savefig(str(a)+"_"+str(b)+"_"+str(c)+"_cluster_"+str(cluster)+
           # ".png", dpi=fig.dpi)


