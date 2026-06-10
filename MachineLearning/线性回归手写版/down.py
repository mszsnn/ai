import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 数据读取
origin = pd.read_csv('home_data_realistic.csv')



# 长度 1000
length = len(origin)

# 定义梯度下降的核心函数
# 损失函数：f(w,b) = n^-1 * sum( y1 - y2 )^2
# 梯度核心   w = w - alpha * f(w, b) 的偏导数


# 定义迭代次数 count
# 这里只用一个特征：sqft_living

x = origin['sqft_living'].to_numpy()
y = origin['price'].to_numpy()


# 损失函数
def loss(w, b):
    y_pred = w * x + b
    return (1 / length) * np.sum((y - y_pred) ** 2)

# 核心函数
def down( w , b , alpha):
   # dw 偏导数 = 1/n * sum(y_pre - y) * x
   # db = 1/n * sum(y_pre - y)
    y_pred = w * x + b
    dw = (2 / length) * np.sum((y_pred - y) * x)
    db = (2 / length) * np.sum(y_pred - y)
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

w = 0
b = 0
alpha = 1e-8
count = 100

loss_list = []
for i in range(count):
    w, b = down(w ,b , alpha)
    l = loss(w, b)
    loss_list.append(l)
    print('loss', l, 'w', w, 'b', b)

# print(length)

# 绘制点
plt.scatter(x, y, color='red', alpha=0.5)

# 绘制拟合的直线
x_line = np.linspace(0, 6000, 50)
y_line = x_line * w + b
plt.plot(x_line, y_line, color='blue')
plt.show()

# loss 曲线
plt.figure()
plt.plot(loss_list)
plt.show()