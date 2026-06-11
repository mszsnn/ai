# 进行 Zscore-归一化
import numpy as np


def z_score_handler(X):

    # 平均数
    mean = np.mean(X, axis=1, keepdims=True)
    # 加上是为了防止 0
    std = np.std(X, axis=1, keepdims=True) + 1e-8

    x_end = (X - mean) / std

    return x_end, mean, std

