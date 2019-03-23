# 神经网络处理数据
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print("Cancer data per-feature maxima:\n{}".format(cancer.data.max(axis=0)))
# 直接使用数据
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))
# 使用标准化数据
scaler = StandardScaler()
X_s_train = scaler.fit_transform(X_train)
X_s_test = scaler.fit_transform(X_test)

mlp = MLPClassifier(random_state=0)
mlp.fit(X_s_train, y_train)

print("Accuracy on training set: {:.2f}".format(mlp.score(X_s_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_s_test, y_test)))
# 增加迭代次数
mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_s_train, y_train)

print("Accuracy on training set: {:.3f}".format(
    mlp.score(X_s_train, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_s_test, y_test)))
# 加大alpha数值，增加泛化能力
mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_s_train, y_train)

print("Accuracy on training set: {:.3f}".format(mlp.score(X_s_train, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_s_test, y_test)))

# 显示输入层和第一个隐藏层之间的参数
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
plt.show()
