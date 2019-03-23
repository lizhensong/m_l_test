# 线性支持向量机创建高维分割数据。最后等高线展示。
import numpy as np
import matplotlib.pyplot as plt
import mglearn

from mpl_toolkits.mplot3d import Axes3D, axes3d

from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

X, y = make_blobs(centers=4, random_state=8)
y = y % 2
linear_svm = LinearSVC().fit(X, y)

mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

# np.vstack():在竖直方向上堆叠
# np.hstack():在水平方向上平铺
# X[:, 1:]中1:表示从1列开始到后边，还是列。如果没有：列会变成行
X_new = np.hstack([X, X[:, 1:] ** 2])

linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
print(coef)
print(intercept)
figure = plt.figure()
# visualize in 3D
# azim沿着z轴旋转，elev沿着y轴
ax = Axes3D(figure, elev=-152, azim=-26)

# 返回50个均分样本
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
# 将XX行扩 YY列扩 数量级yy个数。这样可以完整考虑每个点。
XX, YY = np.meshgrid(xx, yy)
ZZ = -(coef[0] * XX + coef[1] * YY + intercept)/coef[2]
# rstride	数组行步长
# cstride	数组列步长
# 函数上绘制网格
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)

# plot first all the points with y==0, then all with y == 1
# 得到的是一个布尔型数列
mask = y == 0
# xs，ys	数据点的位置。
# ZS	可以是与xs和 ys长度相同的数组，也可以是将所有点放在同一平面中的单个值。默认值为0。
# zdir	在绘制2D集时，使用哪个方向作为z（'x'，'y'或'z'）。
# s	分数大小^ 2。它是一个标量或与x和y长度相同的数组。
# C	一种颜色。
# cmap调用mglearn美化
# edgecolor边缘颜色
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
           cmap=mglearn.cm2, s=60, edgecolor='k')
# x[,]左边是行范围，右边是列范围
# ~这是非
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
            cmap=mglearn.cm2, s=60, edgecolor='k')
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")
plt.show()

# ** 为乘方
ZZ = YY ** 2
# np.c_左右相加（这变为了3列的数组）  np.r_上下相加（这变为一个一维长数组）
# ravel将数组转为一维数组（横的）
# decision_function返回的是里边各个实例到超平面的距离
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
# contourf是画等高线的函数
# dec.reshape(XX.shape)是dec变为XX的矩阵样式
# 前三个参数为x,y对应z。 levels一个数为等高线个数。数字为等高线绘制位置。
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
             cmap=mglearn.cm2, alpha=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
