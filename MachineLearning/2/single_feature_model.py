from pyexpat import features

import pandas as pd
import matplotlib.pyplot as plt   # 绘图、可视化
from sklearn.linear_model import LinearRegression as lr   # 机器学习
import numpy as np   # 数学计算能力
from sklearn.metrics import mean_squared_error

# 默认 pyplot 不识别中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # mac 常用中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题




# 读取数据
df = pd.read_csv('home_data_realistic.csv')

# 简单的一些获取数据



print(df.columns)

features_single = ['sqft_living']
features_multi = ['sqft_living', 'bedrooms', 'waterfront']

# 随机分割
train = df.sample(frac = 0.7, random_state= 42)

# 读取剩余数据
test = df.drop(train.index)

x_train = train[features_single]
y = train['price']


model = lr()
model.fit(x_train,y)


sample = df.iloc[[0]]  # 取一条数据（DataFrame）

print('df.iloc[0]', df.iloc[0])   #  返回的是类型是  dtype: object
print('df.iloc[[0]]', df.iloc[[0]])  # 返回类型是 [1 rows x 21 columns]  二维的， 因为传入的是list

# pandas 的函数预测，采纳都是二维的

pred = model.predict(sample[['sqft_living']])
print("预测价格:", pred[0])

# 在图中画出来， scatter 散点
plt.scatter(x_train, y, alpha = 0.3, label = '测试')

x_line = pd.DataFrame({
    'sqft_living': sorted(train['sqft_living']),
})

y_line = model.predict(x_line)

# plot 画出连续的点
plt.plot(x_line, y_line, color = 'red', label = '回归线')

plt.xlabel('sqft_living 面积大小')
plt.ylabel('price 价钱')
plt.show()


# 计算损失， 也就是误差
x_test  = test[['sqft_living']]
y_test = test['price']
y_pred = model.predict(x_test)
loss = np.mean((y_test - y_pred) ** 2)
# 用模型计算

loss2 = mean_squared_error(y_test, y_pred)


print('loss: ', loss)
print('loss2: ', loss2)


# 用多特征

model_multi = lr()
x_train_multi = train[features_multi]
model_multi.fit(x_train_multi, y)

# 计算损失

y_model_mutil_predict =  model_multi.predict(x_train_multi);
y_model_mutil_loss = mean_squared_error(y_model_mutil_predict, y)


print('loss: ', y_model_mutil_loss)


# 假设一组数据， 我们来实际看看预测价格

act_data = pd.DataFrame({
    'sqft_living': [1630],
    'bedrooms': [2],
    'waterfront': [0],
})

act_y = model_multi.predict(act_data)

print('act_y: ', act_y)