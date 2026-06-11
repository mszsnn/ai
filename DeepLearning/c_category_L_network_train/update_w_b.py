# ============== 5. 扣工资 (参数更新) ==============
import numpy as np


def  update_parameters_deep(parameters, dcache, v, s ,total_steps_index, learn_rate, beta_v = 0.9, beta_s = 0.999):
    """
    深度更新学习率

    w = w - learn_rate * dcache

    parameters [w1, b1, w2, b2....]
    dcache [dw1, db1, dw2, db2....]  这个里面还有 da1 da2 ...
    """

    L = len(parameters) // 2

    for l in range(1, L + 1):

        # 原始不加动量梯度 ==================
        # parameters['W' + str(l)] = parameters['W' + str(l)] - learn_rate * dcache['dW' + str(l)]
        # parameters['b' + str(l)] = parameters['b' + str(l)] - learn_rate * dcache['db' + str(l)]

        # 1. 计算当前的移动速度 (包含 90% 的历史惯性 + 10% 的当前坡度)    新增动量梯度下降
        # v['dW' + str(l)] = beta_v * v['dW' + str(l)] + (1 - beta_v) * dcache['dW' + str(l)]
        # v['db' + str(l)] = beta_v * v['db' + str(l)] + (1 - beta_v) * dcache['db' + str(l)]
        # parameters['W' + str(l)] = parameters['W' + str(l)] - learn_rate * v['dW' + str(l)]
        # parameters['b' + str(l)] = parameters['b' + str(l)] - learn_rate * v['db' + str(l)]


        # 添加 RMSprop 让梯度加速或者减速 ==================

        # v['dW' + str(l)] = beta_s * v['dW' + str(l)] + (1 - beta_s) * dcache['dW' + str(l)] ** 2
        # v['db' + str(l)] = beta_s * v['db' + str(l)] + (1 - beta_s) * dcache['db' + str(l)] ** 2
        #
        # # 偏差修正
        # v_corrected = v['dW' + str(l)] / ( 1 - beta_s ** total_steps_index )
        # b_corrected = v['db' + str(l)] / ( 1 - beta_s ** total_steps_index )
        #
        # parameters['W' + str(l)] = parameters['W' + str(l)] - learn_rate * ( dcache['dW' + str(l)] / np.sqrt(v_corrected + 1e-8))
        # parameters['b' + str(l)] = parameters['b' + str(l)] - learn_rate * ( dcache['db' + str(l)] /  np.sqrt(b_corrected + 1e-8))

         # 添加 Adam 结合两者
        v['dW' + str(l)] = beta_v * v['dW' + str(l)] + (1 - beta_v) * dcache['dW' + str(l)]
        v['db' + str(l)] = beta_v * v['db' + str(l)] + (1 - beta_v) * dcache['db' + str(l)]
        s['dW' + str(l)] = beta_s * s['dW' + str(l)] + (1 - beta_s) * dcache['dW' + str(l)] ** 2
        s['db' + str(l)] = beta_s * s['db' + str(l)] + (1 - beta_s) * dcache['db' + str(l)] ** 2

        vw_corrected = v['dW' + str(l)] / ( 1 - beta_v ** total_steps_index )
        vb_corrected = v['db' + str(l)] / ( 1 - beta_v ** total_steps_index )

        sw_corrected = s['dW' + str(l)] / ( 1 - beta_s ** total_steps_index )
        sb_corrected = s['db' + str(l)] / ( 1 - beta_s ** total_steps_index )

        parameters['W' + str(l)] = parameters['W' + str(l)] - learn_rate * ( vw_corrected / np.sqrt(sw_corrected + 1e-8))
        parameters['b' + str(l)] = parameters['b' + str(l)] - learn_rate * ( vb_corrected /  np.sqrt(sb_corrected + 1e-8))

    return parameters, v, s

