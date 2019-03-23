# k近邻算法
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
iris_dataset = load_iris()
# print(iris_dataset['target_names'])
# print(iris_dataset['feature_names'])
# print(iris_dataset['data'])
# print(iris_dataset['target'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train(创建散点图矩阵)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8, cmap=mglearn.cm3)
# 第一个参数 dataframe：iris_dataframe 按行取数据
# 第二个参数 c=c=y_train 颜色，用不同着色度区分不同种类
# 三：figsize=(15,15) 图像区域大小，英寸为单位
# 四：marker=‘0’ 点的形状，0是圆，1是￥
# 五： hist_kwds={‘bins’:50} 对角线上直方图的参数元组
# 六：s=60 描出点的大小
# 七：alpha=.8 图像透明度，一般取(0,1]
# 八：cmap=mglearn.cm3 mylearn实用函数库，主要对图进行一些美化等私有功能，可见
plt.show()
# 训练
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
# 测试
print(knn.score(X_test, y_test))
# 预测
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:",
       iris_dataset['target_names'][prediction])
