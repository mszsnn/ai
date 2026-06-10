import numpy as np
import matplotlib.pyplot as plt


def zscore_normalize(X):
    mean = np.mean(X, axis=0)  # 平均数
    std = np.std(X, axis=0) # 标准差
    X_norm = ( X - mean ) / std  # 缩放
    return X_norm, mean, std



def sigmoid(x):
    """
        纯粹的 Sigmoid 数学公式：1 / (1 + e^-z)
    """
    return 1 / (1 + np.exp(-x))



def compute_cost_logistic(X, y, w, b):
    """
        纯粹的交叉熵数学公式： -y*log(f) - (1-y)*log(1-f)
    """

    m = X.shape[0]

    z_prev =  X @ w + b #  这里单独拿出来是对应线性回归 perv

    y_prev = sigmoid(z_prev) # 这是真正的预测值

    loss = -y * np.log(y_prev) - (1 - y) * np.log(1 - y_prev)

    return np.sum(loss) / m


def compute_dj(X, y, w, b):
    """
    纯粹的梯度求导结果：与线性回归一字不差！
    """

    m = X.shape[0]

    f_wb = sigmoid(X @ w + b)  # 预测值

    err = f_wb - y  # 误差 m X 1 矩阵

    # （预测值 - 实际值）* 特征 的和 刚好就是 矩阵 X 的转置矩阵 @ 误差
    dj_dw =  (X.T @ err) / m
    dj_db = np.sum(err) / m

    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, iterations):
    w = np.copy(w_in)
    b = b_in
    J_history = []
    for i in range(iterations):
        dw, db = compute_dj(X, y, w, b)

        w = w - alpha * dw
        b = b - alpha * db

        cost = compute_cost_logistic(X, y, w, b)

        J_history.append(cost)
        print(f"迭代 {i:4d}: 交叉熵 Cost = {cost:10.4f}")

    return w, b, J_history


X_train = np.array([
    [1.2, 35], [1.5, 42], [1.8, 38], [2.0, 50],
    [4.5, 65], [5.0, 70], [5.5, 58], [6.2, 80]
])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

X_norm, mu, sigma = zscore_normalize(X_train)

w_init = np.zeros(X_norm.shape[1])
b_init = 0.
alpha = 0.5
iterations = 100

w_final, b_final, J_hist = gradient_descent(X_norm, y_train, w_init, b_init, alpha, iterations)

print(f"\n训练完成！完美参数: w = {w_final}, b = {b_final}")

plt.plot(J_hist, color='red')
plt.title("Pure Math: Cross-Entropy Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.grid(True)
plt.show()