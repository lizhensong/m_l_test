# 高斯核模拟和参数C和gamma的选择
import matplotlib.pyplot as plt
import mglearn

from sklearn.svm import SVC

X, y = mglearn.tools.make_handcrafted_dataset()
# kernel为选择核参数，rbf为径向基函数（高斯核）(为默认)
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
# 精度
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# 支持向量机的支持向量
sv = svm.support_vectors_
# dual_coef_决策函数中支持向量的系数
sv_labels = svm.dual_coef_.ravel() > 0
# markeredgewidth标记边缘宽度,s标记大小
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

# 参数C和gamma的选择对拟合度的影响
# 大 高拟合，低泛化（高方差，低偏差）
# 小 低拟合，高泛化（低方差，高偏差）
fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)

axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"],
                  ncol=4, loc=(.9, 1.2))
plt.show()
