import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def zscore_normalize_features(X):
    # 计算每一列的均值 (axis=0)
    mean = np.mean(X, axis=0)

    # 计算每一列的标准差 (axis=0)
    std = np.std(X, axis=0)

    X_norm = (X - mean)/std

    # 执行 (x - mu) / sigma
    return X_norm, mean, std


# =====================================================================
# 2. 向量化计算核心 (Vectorized Implementation)
# =====================================================================

# 计算 损失函数
def compute_cost(X, y, w, b):
    # X 向量 （m,n）
    # w 权重向量 （n,）
    m = X.shape[0]

    # 全部的预测值
    prev_value = X @ w + b

    # 损失函数
    cost = np.sum((prev_value - y) ** 2)

    return cost / (2 * m)


# 计算梯度
# 梯度的核心是 偏导 误差值 * 特征值 的平均

def compute_gradient(X, y, w, b):

    m, n = X.shape
    # 算出所有误差， 这里这么做的原因是  矩阵乘法 就是矩阵的每一行和 向量 做点积

    err = X @ w + b - y   # 得到一个 m X 1 的误差矩阵

    # 误差值 * 特征值 的平均 X.T 其实就是原来 m * n 矩阵 变成 n * m
    # 然后  n X m  m X 1 矩阵乘法得到的结果是 n X 1 刚好就是 n 个 特征 每个特征的偏导
    dj_dw = (X.T @ err) / m
    dj_db = np.sum(err) / m

    return dj_dw, dj_db



# 训练

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    w = w_in
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

        if i % (num_iters // 10) == 0:
            print(f"迭代 {i:4d}: Cost = {cost:10.4f}")

    return w, b, J_history





# ==========================================
# 1. 模拟读取真实的表格数据 (DataFrame)
# ==========================================
# 在真实场景中，这行代码通常是：df = pd.read_csv('california_housing.csv')
# 这里我们用字典模拟从文件中读取了一个包含 5 条数据的表格
raw_data = {
    'Area(sqft)': [2104, 1416, 852, 1930, 1200, 1534, 1000, 2400],  # 面积
    'Bedrooms':   [5, 3, 2, 4, 3, 3, 2, 4],                # 卧室数
    'Floors':     [1, 2, 1, 2, 1, 2, 1, 2],   # 楼层数
    'Age(years)': [45, 40, 35, 15, 20, 30, 10, 5],     # 房龄
    'Price(1000$)':[460, 232, 178, 315, 200, 270, 190, 500]  # 真实的售价 (标签)
}


df = pd.DataFrame(raw_data)

X = df.drop('Price(1000$)', axis=1).to_numpy()
y = df['Price(1000$)'].to_numpy()
# 执行缩放
X_norm, avg, std = zscore_normalize_features(X)
print(f"数据缩放完成。\n均值: {avg}\n标准差: {std}\n")

# 初始化训练
alpha = 0.1
iterations = 100
w_init = np.zeros(X.shape[1])
b_init = 0.


w_final, b_final, J_hist = gradient_descent(X_norm, y, w_init, b_init, alpha, iterations)



# 预测新数据
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - avg) / std

# 【修改点 4】：使用 @ 替代 np.dot
prediction = x_house_norm @ w_final + b_final

print(f"\n新房预测价格: ${prediction:.2f}k")


# 绘制学习曲线
plt.plot(J_hist)
plt.title("Learning Curve (Vectorized with @ & Scaled)")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()