
import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention 是为了添加 全局的语义理解， 让 token 和 token 之间进行交流
# 前馈神经网络， 只是对 token 本身的特征进行深度加工， 经过两个线性层和一个非线性层

class FeedforwardLayer(nn.Module):
    # 核心公式 FFN(x) = ReLU(x * W1 + b1) * W2 + b2

    def __init__(self, d_model,  high_dimension, dropout = 0.1):
        super().__init__()

        # 线性变换到高维度的映射
        self.to_high = nn.Linear(d_model, high_dimension)

        # 线性变换到原有维度的映射
        self.to_old = nn.Linear(high_dimension, d_model)

       #  0.1 代表 10% 的概率设置 设置为 0， 防止过拟合， 其实就是将里面的每个元素， 进行随机
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x [batch_size, sentence_len, d_model]
        first_layer = F.relu(self.to_high(x))

        first_layer = self.dropout(first_layer)

        # 重新线性映射回去原来的维度
        end_result = self.to_old(first_layer)

        return end_result
