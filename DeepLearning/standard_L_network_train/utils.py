# 辅助函数
import numpy as np
import math
def get_mini_batches(X, Y, batch_size=64):
    """
    将全量数据 (X, Y) 随机打乱并切分成指定大小的 Mini-batch。

    参数:
    X -- 输入数据，维度为 (特征数量 n_x, 样本数量 m)
    Y -- 真实标签，维度为 (1, 样本数量 m)
    mini_batch_size -- 每个小包的样本数量，通常设为 64, 128, 256

    返回:
    mini_batches -- 一个包含 (mini_batch_X, mini_batch_Y) 元组的列表
    """

    m = X.shape[1]
    mini_batches = []

    # 1 随机打乱， 生成 0 - (m -1) 的随机数列
    permutation = np.random.permutation(m)

    # 行不变，Y 按照索引进行随机打乱
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]


    steps = m // batch_size
    for i in range(steps):
        X_batch = X_shuffled[:, i * batch_size:(i + 1) * batch_size]
        Y_batch = Y_shuffled[:, i * batch_size:(i + 1) * batch_size]

        mini_batches.append((X_batch, Y_batch))


    if m % batch_size != 0:
        X_end =  X_shuffled[:, steps * batch_size:]
        Y_end = Y_shuffled[:, steps * batch_size:]
        mini_batches.append((X_end, Y_end))

    return mini_batches







# 学习率衰减

def alpha_iteration_by_exponent(current_epoch, ini_alpha, decay_rate):
    """
    计算指数衰减的动态学习率。

    返回:
    - 当前轮数应使用的学习率
    """


    return  ini_alpha * (decay_rate ** current_epoch)




def alpha_iteration_by_cos(current_epoch, total_epoch, warmup_steps, max_steps = 1e-3, min_steps = 1e-6):
    """
        计算带有预热期的余弦衰减学习率 (大模型标配)。

        参数:
        - current_epoch: 当前的训练步数 (或 Epoch 数)
        - total_epoch: 整个训练过程的总步数
        - warmup_steps: 用于预热的步数 (通常占总步数的 5% ~ 10%)
        - max_alpha: 预热结束时达到的峰值学习率 (默认 0.001)
        - min_alpha: 训练结束时的谷底学习率 (默认 1e-6)

        返回:
        - 当前步数应使用的学习率
        """

    # 阶段1

    if current_epoch < warmup_steps:
        return current_epoch / warmup_steps * max_steps

    # 阶段2

    process = (current_epoch - warmup_steps) / (total_epoch - warmup_steps)

    cos_value = 1 + math.cos(math.pi * process)

    return min_steps + 0.5 * (max_steps - min_steps) * cos_value