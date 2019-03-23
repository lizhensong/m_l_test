# 高斯模型，数据处理
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

svc = SVC()
svc.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))

# 画箱体图 X_train为数据集，
# manage_xticks是否自动调整xticks（刻度、名字）和xlim（范围）的值
plt.boxplot(X_train, manage_xticks=False)
# y轴刻度类型
plt.yscale("symlog")
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.show()

# 计算每一列最小值
# axis为0，矩阵按列
# axis为1，矩阵按行
min_on_training = X_train.min(axis=0)
# 计算每个特征的范围
range_on_training = X_train.max(axis=0)-min_on_training
# 减去最小值再除范围
# 最后特征会在0~1之间
X_train_scaled = (X_train - min_on_training) / range_on_training
print("Minimum for each feature\n", X_train_scaled.min(axis=0))
print("Maximum for each feature\n", X_train_scaled.max(axis=0))
# 对训练集的数据进行相同处理
X_test_scaled = (X_test - min_on_training) / range_on_training

svc1 = SVC()
svc1.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(svc1.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc1.score(X_test_scaled, y_test)))

svc2 = SVC(C=1000)
svc2.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(svc2.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc2.score(X_test_scaled, y_test)))