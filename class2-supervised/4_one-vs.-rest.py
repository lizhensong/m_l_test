# 使用线性SVM完成 一对余 的三分类
import numpy as np
import matplotlib.pyplot as plt
import mglearn

from sklearn.svm import LinearSVC

from sklearn.datasets import make_blobs

# make_blobs会根据用户指定的特征数量、中心点数量、范围等来生成几类数据，这些数据可用于测试聚类算法的效果
# n_samples是待生成的样本的总数。
# n_features是每个样本的特征数。
# centers表示类别数。
# cluster_std表示每个类别的方差，例如我们希望生成2类数据，其中一类比另一类具有更大的方差，可以将cluster_std设置为[1.0,3.0]。
X, y = make_blobs(random_state=42)

linear_svm = LinearSVC().fit(X, y)
print("Coefficient shape: ", linear_svm.coef_.shape)
print("Intercept shape: ", linear_svm.intercept_.shape)

# 画出这个向量机的预测范围
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
# 画出所有训练集的位置，y不同点样式不同
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# 返回-15到15之间均匀分布的样本
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  mglearn.cm3.colors):
    # 画函数 其方程为-（x*w[0]+b）/w[1]
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
            'Line class 2'], loc=(1.01, 0.3))
plt.show()
