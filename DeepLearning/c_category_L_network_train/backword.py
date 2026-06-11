import numpy as np

from activation_function_and_backward import softmax_backward, relu_backward

def calc_backward(Y, parameters, cache, lambda_ = 0):
    """
         反向传播 (动静分离版)
         - Y: 真实标签
         - cache: 员工草稿纸 (只包含动态的 Z1, A1, Z2, A2...)
         - parameters: 公司固定资产 (只包含静态的 W1, b1, W2, b2...)
    """
    L = len(parameters) // 2
    m = Y.shape[1]
    dcache = {}

    AL = cache["A" + str(L)]
    WL = parameters['W' + str(L)]
    prev_A = cache["A" + str(L - 1)]


    # ================= 2. 第 L 层的反向追责 (Sigmoid) =================
    dZL = softmax_backward(AL, Y)
    dWL = (1 / m) * np.dot(dZL, prev_A.T) + (lambda_ / m) * WL
    dbL = (1 / m) * np.sum(dZL, axis=1, keepdims=True)

    dcache["dW" + str(L)] = dWL
    dcache["db" + str(L)] = dbL
    # 本层的转置矩阵 * 本层的责任
    dcache['dA' + str(L - 1)] = np.dot(WL.T, dZL)


    for l in reversed(range(1, L)):

        # ================= 2. 第 l 层的反向追责 (relu) =================
        Z_tmp = cache["Z" + str(l)]
        prev_A_tmp = cache["A" + str(l - 1)]
        w_tmp = parameters['W' + str(l)]

        dAl_tmp = dcache["dA" + str(l)]
        dZl = relu_backward(dAl_tmp, Z_tmp)

        dWl = (1 / m) * np.dot(dZl, prev_A_tmp.T) + (lambda_ / m) * w_tmp
        dbl = (1 / m) * np.sum(dZl, axis=1, keepdims=True)

        dcache["dW" + str(l)] = dWl
        dcache["db" + str(l)] = dbl

        dcache['dA' + str(l - 1)] = np.dot(w_tmp.T, dZl)


    return dcache
















