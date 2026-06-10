# 模块一：架构师下发编制（初始化参数）只要你传入一个列表
# （比如 layer_dims = [2, 4, 1]），
# 这个函数就会自动用维度法则把所有的 W 和 b 矩阵造出来。



import numpy as np


def init_parameters(layer_dims):
    """
    layer_dims: 包含每一层打工人数的列表。
    比如 [2, 5, 3, 1] 表示：输入特征2，第一层5人，第二层3人，输出层1人。
    """
    np.random.seed(2)

    # 但是实际是 l - 1 层
    l = len(layer_dims)

    # 我们其实不需要 W0 d0
    parameters = {}
    for i in range(1, l):
        # parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01

        # 后续优化代码
        # 这里的优化， 主要是为了防止梯度爆炸和梯度消失，
        # 下面这行代码 *  np.sqrt(2.0 / layer_dims[i - 1]) 适合 ReLU  的激活函数
        # 下面这行代码 *  np.sqrt(1.0 / layer_dims[i - 1]) 适合 tanh  的激活函数

        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(2.0 / layer_dims[i - 1])
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

        # 断言判定维度正确， 如果不正确，直接报错
        assert parameters['W' + str(i)].shape == (layer_dims[i], layer_dims[i - 1])
        assert parameters['b' + str(i)].shape == (layer_dims[i], 1)


    return parameters

# 初始化动量梯度下降的初始值， 动量的初始值肯定和 w b 是同一个维度

def initialize_velocity(parameters):
    """
    初始化速度字典 v，包含每个参数的当前速度。
    """

    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(1, L + 1):
        # 巧妙使用 np.zeros_like，直接生成和 W, b 形状一样但全为 0 的矩阵
        v['dW' + str(l)] = np.zeros_like(parameters['W' + str(l)])
        v['db' + str(l)] = np.zeros_like(parameters['b' + str(l)])

        s['dW' + str(l)] = np.zeros_like(parameters['W' + str(l)])
        s['db' + str(l)] = np.zeros_like(parameters['b' + str(l)])
    return v, s






if __name__ == '__main__':
    parameters = init_parameters(layer_dims=[2, 4, 1])

    print(len(parameters) // 2)
    print(parameters)


