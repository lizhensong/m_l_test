import numpy as np
import matplotlib.pyplot as pl
x = np.linspace(-10, 10, 100)  # 在-10和10之间生成100个数
y = np.sin(x)  # 用正弦函数创建第二个数组
print(x)
print(y)
pl.plot(x, y, marker="o")  # 绘制x,y图像 marker是显示点样式
pl.show() 