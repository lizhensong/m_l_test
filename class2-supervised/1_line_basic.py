import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
h = lr.coef_*X +lr.intercept_
plt.plot(X,h)
# plt.plot(x, y, 'xxx', label=, linewidth=)
#
# 参数1：位置参数，点的横坐标，可迭代对象
#
# 参数2：位置参数，点的纵坐标，可迭代对象
#
# 参数3：位置参数，点和线的样式，字符串
# 此参数分三部分，点线的颜色、点的形状、线的形状
# 点线的颜色：g | green；b | blue；c | cyan 蓝绿色；m - magenta 品红色 ...
# 点的形状：. | 点儿；v | 实心倒三角；o | 实心圆；* | 实心五角星；+ | 加号 ...
# 线的形状：- | 实线；-- | 虚线；-. 点划线
#
# 参数 4：label 关键字参数，设置图例，需要调用 plt 或子图的 legend 方法
#
# 参数 5：linewidth 关键字参数，设置线的粗细
plt.plot(np.matrix(X),y ,"o")
plt.show()
print(X)
print(y)
print("lr.coef_:", lr.coef_)  # 是一个数组，每个元素对应一个特征。
print("lr.intercept_:", lr.intercept_)  # 是一个浮点数，为偏移量。
print(lr.score(X_test, y_test))  # 测试精度。
