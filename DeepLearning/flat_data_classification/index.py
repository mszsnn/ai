import numpy as np

# 单隐藏层神经网络
# 用纯 numpy 来实现

def sigmoid(z):
    """
    Sigmoid激活函数 (员工B的免责阀门)
    """
    return 1 / (1 + np.exp(-z))


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



def layer_sizes(X, Y):
    """
    获取神经网络的三围维度
    """
    n_x = X.shape[0] # 输入层的大小 2
    n_y = Y.shape[0] # 输出层大小 1

    return n_x, n_y


def initialize_parameters(n_x, n_h, n_y):
    """
    随机初始化权重！(打破对称性陷阱，并乘以0.01防止梯度消失)
    """
    # 保证每次随机的结果是一样的
    np.random.seed(2)

    # W1: 隐层权重，维度 (n_h, n_x)  n_h 个神经元 n_x 个特征
    # n_x 输入的特征数  n_h 神经元的数量  x_y 输出的神经元数量
    W1 = np.random.randn(n_h, n_x) * 0.01
    # B1 的行数一定是和 W1 的神经元个数是一样的
    B1 = np.zeros((n_h, 1))

    W2 = np.random.randn(n_y, n_h) * 0.01

    B2 = np.zeros((n_y, 1))

    parameters = {"W1": W1, "B1": B1, "W2": W2, "B2": B2}

    return parameters




def forward_propagation(X, parameters):
    """
    前向传播 (从左到右执行计算图)
    """
    W1 = parameters["W1"]
    B1 = parameters["B1"]
    W2 = parameters["W2"]
    B2 = parameters["B2"]


    # 第一层
    Z1 = np.dot(W1, X) + B1
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + B2
    A2 = sigmoid(Z2)

    caches = { "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2 }
    return A2, caches


def compute_cost(A2, Y):
    """
    计算交叉熵损失 (大老板的怒火总量)
    """
    # -[y * log(a) + (1 - y) * log(1 - a)]
    m = A2.shape[1]

    a_cost = -1 * ( Y * np.log(A2) + (1 - Y) * np.log(1 - A2) )

    cost = np.sum(a_cost) / m

    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    反向传播 (从右到左，用维度碰撞法推导的终极公式)
    # ---------------- 核心推导落地 ----------------
    # 1. 第二层误差 (预测值 - 真实值)
    dz2 = A2 - Y
    所有人都犯错， 那就是体制的问题
    db2 = 1/m * dz2

    上级的误差 * 输入的音量， 就是特征的导数
    dw2 = 1/m * dz2 @ A1.T

    给多个领导干活，所以这里需要所有相关的总和， 所以需要转置
    连带责任 * 自身免责条款 （激活函数的导数）
    dz1 = w2.T @ dz2 * g 的导数 （z1）
    db1 = 1/m * dz1
    """
    m = X.shape[1]
    A2 = cache["A2"]
    A1 = cache["A1"]
    W2 = parameters["W2"]

    # 终极误差
    dz2 =  A2 - Y

    # 终极误差 @ 输入的音量
    dw2 = (1 / m) * dz2 @ A1.T

    # 压缩行 ， 求行方向的和
    db2 = (1 / m) * np.sum(dz2, axis = 1, keepdims=True)

    # 3. 跨层甩锅给第一层 (W2转置 乘以 dZ2，再乘以本层的 Tanh 导数)
    # Tanh(z) 的导数是(1 - a ^ 2)
    dz1 =( W2.T @ dz2) * (1 - np.power(A1, 2))

    dw1 = (1 / m) * dz1 @ X.T

    db1 = (1 / m) * np.sum(dz1, axis = 1, keepdims=True)

    dw_map = { 'dw1': dw1, 'dw2': dw2, 'db2': db2, 'db1': db1 }
    return dw_map

def update_parameters(parameters, dw_map, study_rate = 1.2):
    """
    梯度下降，更新参数 (扣工资调整策略)
    """

    W1 = parameters["W1"] - study_rate * dw_map["dw1"]
    B1 = parameters["B1"] - study_rate * dw_map["db1"]
    W2 = parameters["W2"] - study_rate * dw_map["dw2"]
    B2 = parameters["B2"] - study_rate * dw_map["db2"]

    parameters = { "W1": W1, "B1": B1, "W2": W2, "B2": B2 }

    return parameters



# ==========================================
# 第三部分：组装流水线引擎
# ==========================================


def pipe_line(X, Y, n_h, train_count = 10000):
    """
    终极黑盒：把所有零件组装成一个完整的神经网络训练循环
    """
    # 计算维度
    n_x, n_y = layer_sizes(X, Y)

    # 得到初始化的参数
    parameters = initialize_parameters(n_x, n_h, n_y)

    # 训练

    for i in range(train_count):
        # 步骤1 前向传递
        A2, cache = forward_propagation(X, parameters)

        # 计算总体损失
        cost = compute_cost(A2, Y)

        # 计算往回算责任
        dw_map = backward_propagation(parameters, cache, X, Y)

        # 更新梯度
        parameters = update_parameters(parameters, dw_map)

        print(f"迭代次数 {i}: 损失值 Cost = {cost:.6f}")

    return parameters



X, Y = load_planar_dataset()
print(f"数据加载完成！输入 X 维度: {X.shape}, 标签 Y 维度: {Y.shape}")

n_h = 4
print(f"\n开始训练包含 {n_h} 个神经元的单隐层神经网络...")

parameters = pipe_line(X, Y, n_h)


