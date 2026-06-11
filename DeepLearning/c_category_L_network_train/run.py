
import numpy as np
from main import deep_neural_network_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils import convert_to_one_hot
from z_store import z_score_handler

# 此文件夹下面的所有文件，都是基于 2分类 standard_L_network_train 修改而来


def load_iris_dataset():
    """
    加载鸢尾花真实数据集，并转换为吴恩达标准维度
    X: (n_x, m)  特征矩阵
    Y: (C, m)    One-hot标签矩阵
    """
    # 1. 加载真实数据
    iris = datasets.load_iris()
    X_raw = iris.data  # 形状 (150, 4)
    Y_raw = iris.target  # 形状 (150,)

    # 2. 划分训练集和测试集 (80% 训练, 20% 测试)
    X_train, X_test, Y_train, Y_test = train_test_split(X_raw, Y_raw, test_size=0.2, random_state=42)

    # 3. 🚨 架构师核心操作：维度转置对齐 🚨
    # sklearn 默认是 (样本数, 特征数)，我们要转成 (特征数, 样本数)
    X_train = X_train.T  # 变成 (4, 120)
    X_test = X_test.T  # 变成 (4, 30)

    X_train, mu, sigma = z_score_handler(X_train)
    X_test = (X_test - mu) / sigma  # 注意：测试集必须用训练集的 mu 和 sigma！


    # 5. 把标签转换为 One-Hot 矩阵
    C = 3  # 鸢尾花有 3 个品种
    Y_train_one_hot = convert_to_one_hot(Y_train, C)  # 变成 (3, 120)
    Y_test_one_hot = convert_to_one_hot(Y_test, C)  # 变成 (3, 30)

    print(f"数据加载完毕！")
    print(f"X_train 维度: {X_train.shape} (特征数={X_train.shape[0]}, 样本数={X_train.shape[1]})")
    print(
        f"Y_train_one_hot 维度: {Y_train_one_hot.shape} (类别数={Y_train_one_hot.shape[0]}, 样本数={Y_train_one_hot.shape[1]})")

    return X_train, Y_train_one_hot, X_test, Y_test_one_hot

# 测试一下
X_train, Y_train, X_test, Y_test = load_iris_dataset()

features = X_train.shape[0]
types = Y_train.shape[0]

print(features, types)


deep_neural_network_model(X_train, Y_train, [features, 10, 10, types], [0.001, 0.01], 1000, 0.1 )