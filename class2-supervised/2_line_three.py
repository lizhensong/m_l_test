# 线性回归三种：普通线性回归、岭回归、lasso回归
import numpy as np
import mglearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# 获取数据
X, y = mglearn.datasets.load_extended_boston()
# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("普通线性回归")
# 普通线性回归算法（无正则化）
lr = LinearRegression().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

print("岭回归")
# 岭回归算法（普通线性加入正则化L2，正则化参数默认为1）
ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

# 将正则化参数设为10，减少拟合能力，增加泛化能力（提高偏差，降低方差）
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

# 将正则化参数设为0.1，减少泛化能力，增加拟合能力（提高方差，降低偏差）
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))

print("lasso回归")
# lasso回归算法（普通线性加入正则化L1，正则化参数默认为1，会自动忽略一些不重要的特征）
lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso.coef_ != 0))

# 将正则化参数设为0.01，参数降低，特征值数量保留变多。减少泛化能力，增加拟合能力（提高方差，降低偏差）
# "max_iter"运行迭代的最大次数
# otherwise the model would warn us that we should increase max_iter.
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso001.coef_ != 0))

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso00001.coef_ != 0))

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
# 获取当前x轴左，右数值
xlims = plt.xlim()
# 画一条横线第一个参数y坐标，第二个x起点，第三个x终点
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-25, 25)
# 自动获取签名
plt.legend()
plt.show()
# 这个学习曲线展示
mglearn.plots.plot_ridge_n_samples()
plt.show()

plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")

plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
# ncol表示图例展示列数，loc表示图例显示位置
# X,y用于定位图例，也可用单键词"bottomright", "bottom", "bottomleft", "left", "topleft", "top", "topright", "right" and "center"代替
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.show()
