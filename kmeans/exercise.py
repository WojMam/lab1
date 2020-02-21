'''
    Authors:
    Zuzanna Molęda
    Sebastian Tomaszewski
    Wojciech Mamys

    Informatyka Stosowana
    II rok, III semestr
'''
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
from matplotlib import cm
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples

train_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00396/Sales_Transactions_Dataset_Weekly.csv"
train = pd.read_csv(train_url)
data_source = train[train.columns[1:]]

'''KMeans and KMeans++'''
cluster_size = int(input("Number of clusters: "))
k_means_pp = cluster.KMeans(n_clusters=cluster_size)
kmeans_model_pp = k_means_pp.fit(data_source)
k_means = cluster.KMeans(n_clusters=cluster_size, init='random')
kmeans_model = k_means.fit(data_source)
centers_kmeans_pp = np.array(kmeans_model_pp.cluster_centers_)
centers_kmeans = np.array(kmeans_model.cluster_centers_)

'''AglomerativeClustering'''
agglomerative = AgglomerativeClustering(linkage="ward", n_clusters=cluster_size).fit(data_source)
linkage_matrix = linkage(data_source, 'ward')
sns.clustermap(linkage_matrix)

'''DBSCAN'''
dbscan = DBSCAN(eps=20, min_samples=3, algorithm='auto').fit(data_source)
dbscan.fit_predict(data_source)
dbscan_labels = labels = dbscan.labels_

plt.figure(figsize=(14, 12))
plt.title('Comparison of unsupervised clustering methods')

plt.subplot(221)
plt.title("K-means++")
plt.scatter(data_source.iloc[:, 0], data_source.iloc[:, 1], c=kmeans_model_pp.labels_, label='Cluster points')
plt.scatter(centers_kmeans_pp[:, 0], centers_kmeans_pp[:, 1], c='r', marker='d', s=100, label='Centroids')
plt.legend(loc='upper left')

plt.subplot(222)
plt.title("K-means")
plt.scatter(data_source.iloc[:, 0], data_source.iloc[:, 1], c=kmeans_model.labels_, label='Cluster points')
plt.scatter(centers_kmeans[:, 0], centers_kmeans[:, 1], c='r', marker='d', s=100, label='Centroids')
plt.legend(loc='upper left')

plt.subplot(223)
plt.title("Agglomerative Clustering")
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',
    p=24,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
)
plt.legend(loc='upper left')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.tight_layout()

plt.subplot(224)
plt.title("DBSCAN")
plt.scatter(data_source.iloc[:, 0], data_source.iloc[:, 1], c=dbscan_labels, label='Cluster points')
plt.legend(loc='upper left')

plt.figure(figsize=(14, 12))
data = train
values = data.get_values()
X = np.array(values)
A = np.delete(X, 0, 1)

km = cluster.KMeans(n_clusters=cluster_size, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)

y_km = km.fit_predict(A)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(A, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals)
    yticks.append((y_ax_upper + y_ax_lower) / 2)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Skupienie')
plt.xlabel('Współczynnik profilu')

plt.show()
