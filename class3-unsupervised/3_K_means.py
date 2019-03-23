# K-means算法
import matplotlib.pyplot as plt
import mglearn

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs

# k-均值聚类算法效果展示
mglearn.plots.plot_kmeans_algorithm()
plt.show()
mglearn.plots.plot_kmeans_boundaries()
plt.show()

# 生成模拟二维数据
X, y = make_blobs(random_state=1)
X_train, X_test = train_test_split(X, random_state=0)
# 分成3部分的模型
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)
# 输出分类后数据的标签 kmeans.labels_
print("Cluster memberships:\n{}".format(kmeans.labels_))
# 预测
print(kmeans.predict(X_test))

mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], kmeans.predict(X_test), markers='*')
mglearn.discrete_scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2],
    markers='^', markeredgewidth=2)
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# using two cluster centers:
kmeans2 = KMeans(n_clusters=2)
kmeans2.fit(X_train)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], kmeans2.labels_, ax=axes[0], markers='o')
mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], kmeans2.predict(X_test), markers='*', ax=axes[0])

# using five cluster centers:
kmeans5 = KMeans(n_clusters=5)
kmeans5.fit(X_train)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], kmeans5.labels_, ax=axes[1], markers='o')
mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], kmeans5.predict(X_test), markers='*', ax=axes[1])
plt.tight_layout()
plt.show()
