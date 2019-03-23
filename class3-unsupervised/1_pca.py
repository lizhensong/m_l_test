# 使用StandardScaler进行数据缩放（标准化）使每个特征方差为1
# 对缩放的数据进行PCA降维处理
import numpy as np
import matplotlib.pyplot as plt
import mglearn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.datasets import load_breast_cancer

# 主成分分析算法图示
mglearn.plots.plot_pca_illustration()
plt.show()

# 查看每个特征和类别关系
cancer = load_breast_cancer()
fig, axes = plt.subplots(15, 2, figsize=(10, 20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

ax = axes.ravel()

for i in range(30):
    # 绘制直方图，第一个参数是数据，bins为统计的区间个数
    # 返回值 前一个数组是直方图的每个区间统计量，后一个数组是区间分割值（有区间个数+1个）
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    # pyplot绘制直方图，第一个参数数据，第二个参数区间
    ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
    ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    # 纵坐标标度为空
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["malignant", "benign"], loc="best")
# 自动调整子图参数，使之填充整个图像区域
fig.tight_layout()
plt.show()

# 使用StandardScaler缩放数据，使每个特征值的方差均为1
scaler = StandardScaler()
# scaler.fit(cancer.data)
# X_scaled = scaler.transform(cancer.data)
# 等价
X_scaled = scaler.fit_transform(cancer.data)

# 保留数据前俩个主成分
pca = PCA(n_components=2)
# pca.fit(X_scaled)
# X_pca = pca.transform(X_scaled)
# 等价
X_pca = pca.fit_transform(X_scaled)

print("Original shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))


# 对俩个主成分作图，按类别着色
# plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()

# 在components_中保存的是生成的主成分和各个特征的关系
print("PCA component shape: {}".format(pca.components_.shape))
print("PCA components:\n{}".format(pca.components_))

# 将其用热图系数可视化
# cmap设定颜色图谱 （翠绿色）
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["First component", "Second component"])
# 添加色标
plt.colorbar()
# ha 确定X轴名字位置
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")
plt.gca().set_aspect("equal")
plt.show()
