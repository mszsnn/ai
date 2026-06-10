import numpy as np

# 为了 深度神经网络服务, 求 Z 或者 dZ

def sigmoid(Z):
    """这里sigmoid 可能会溢出比如 e**1000 """
    A = 1 / (1 + np.exp(-Z))
    return A


def sigmoid_backward(dA, Z):
    """
    Sigmoid的反向破甲公式  直接计算出来 dZ
    激活函数当成了一个独立的反向传播黑盒节点
    """
    s = 1 / (1 + np.exp(-Z))
    d = s * (1 - s)
    dZ = dA * d
    return dZ


def relu(Z):
    """隐藏层的免责阀门：小于0装死，大于0放行"""
    A = np.maximum(0, Z)
    return A


def relu_backward(dA, Z):
    """
    ReLU的反向破甲公式：如果当年Z<=0，这层不背锅(导数为0)
    relu 的导数， 要么 为 1 要么 0
    * dA 也就是 dA 矩阵的部分值，变成 0
    """

    dZ = np.array(dA, copy = True) # 深度拷贝
    dZ[Z <= 0] = 0 # 神奇魔法

    # NumPy会先去看Z矩阵，它会生成一张隐形的“对错表”：如果 $Z$ 里的某个数字 <= 0，就是 True；否则是 False
    # 它拿着这张“对错表”，去覆盖在刚刚复印好的 dZ 矩阵上。凡是对应位置为 True 直接把 dZ 里的数字强行改成 0

    return dZ




