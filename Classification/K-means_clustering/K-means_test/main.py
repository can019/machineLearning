from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1,2],[1,4],[1,0],[10,2],[10,4],[10,0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# kmeans.labels_array([1,1,1,0,0,0],dtype=int32)
kmeans.predict([[0,0],[12,3]])
# array([1, 0], dtype=int32)
kmeans.cluster_centers_
"""
# About KMeans 
KMeans(n_clusters=8, *, init='k-means++',n_init=10,max_iter=300,tol=0.0001,
       precompute_distances='deprecated', verbose=0, random_state=None, copy_x=True,
       n_jobs='deprecated',algorithm='auto')
"""