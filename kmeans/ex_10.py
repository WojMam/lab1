import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

data = pandas.read_csv("Sales_Transactions_Dataset_Weekly.csv")
values = data.get_values()


X = np.array(values)
A = np.delete(X,0,1)

km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300,tol=1e-04,random_state=0)

y_km = km.fit_predict(A)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(A, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals =silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals)
    yticks.append((y_ax_upper + y_ax_lower) /2)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(yticks,cluster_labels + 1)
plt.ylabel('Skupienie')
plt.xlabel('Współczynnik profilu')
plt.show()