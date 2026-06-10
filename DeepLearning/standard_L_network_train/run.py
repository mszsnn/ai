
import numpy as np
from main import deep_neural_network_model


def load_planar_dataset():
    """
    生成一个像“花瓣”一样的非线性平面数据集 (红点和蓝点交织)
    【修改版】：底层数据生成时直接保留两位小数
    X:
    [
        [0.23, 0.34, ...., 400x],
        [0.62, 0.45, ....400y]
    ]

    Y:
    [
        [0, 1, .... 400 类型]
    ]
    """
    np.random.seed(1)
    m = 400
    N = int(m / 2)
    D = 2

    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2

        # 1. 算出原始的超长浮点数坐标矩阵
        raw_coordinates = np.c_[r * np.sin(t), r * np.cos(t)]

        # 2. 【核心修改】使用 np.round 在内存级别直接把数据四舍五入到 2 位小数
        X[ix] = np.round(raw_coordinates, decimals=2)

        Y[ix] = j

    return X.T, Y.T


X, Y = load_planar_dataset()
print(f"数据加载完成！输入 X 维度: {X.shape}, 标签 Y 维度: {Y.shape}")

deep_neural_network_model(X, Y, [2, 20, 7, 5, 1], learning_rate=(0.01, 0.001), num_iterations=500, lambda_= 0.1)