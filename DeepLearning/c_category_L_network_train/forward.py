# 模块二：流水线开动与档案入库（前向传播）这是最美妙的 for 循环。前 L-1 层全用 ReLU，最后一层用 Sigmoid。
import numpy as np
from activation_function_and_backward import relu, softmax

def calc_forward(X, parameters):
    """
        前向传播：X -> [Linear -> ReLU] * (L-1) -> [Linear -> Sigmoid] -> AL
        """

    cache = {} # 缓存
    cache['A0'] = X
    A = X

    L = len(parameters) // 2 # 为啥是  //2 是为了返回整数， /2 总是返回浮点数

    for l in range(1, L):

        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]

        Z = np.dot(W, A) + b

        A = relu(Z)

        cache[ 'Z' + str(l)] = Z
        cache[ 'A' + str(l)] = A

    Z_last = np.dot(parameters['W' + str(L)], A) + parameters['b' + str(L)]
    A_last = softmax(Z_last)

    cache['Z' + str(L)] = Z_last
    cache['A' + str(L)] = A_last

    return A_last, cache





def calc_cost(AL, Y):
    """
        计算多分类交叉熵 Cost
        AL: 预测概率矩阵 (C, m)
        Y: 真实标签的 One-hot 矩阵 (C, m)
        """
    m = Y.shape[1]

    cost = -(1 / m) * np.sum(Y * np.log(AL + 1e-8))
    # 删除长度为1 的维度
    return np.squeeze(cost)



def calc_cost_with_regularization(AL, Y, parameters, lambda_ = 0):
    cross_entropy_cost = calc_cost(AL, Y)
    m = Y.shape[1]
    L = len(parameters) // 2

    result = 0
    for l in range(1, L + 1):
        W = parameters['W' + str(l)]
        result = result + np.sum(np.square(W))

    lambda_value = lambda_ / (2 * m) * result

    # 3. 最终代价 = 正常代价 + 惩罚项
    return cross_entropy_cost + lambda_value