# 多头注意力机制
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model = 512, num_head = 8):
        super().__init__()

        # 确保头数划分合理
        assert  d_model % num_head == 0

        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model // num_head

        # nn.Linear 用来生成一个线性变换， y = x W^t + b, 参数 （输入维度， 输出维度）。仅仅处理最后一个维度

        # nn.Linear(4, 2)  输入 shape [2, 4]  => 变换之后输出  shape [2, 2]

        # query 查询者变换
        self.q_linear_trans = nn.Linear(d_model, d_model)
        self.k_linear_trans = nn.Linear(d_model, d_model)
        self.v_linear_trans = nn.Linear(d_model, d_model)
        self.o_linear_trans = nn.Linear(d_model, d_model)



    def forward(self, q, k, v , mask = None):
        batch_size, sentence_len, _ = q.size()
        k_len = k.size(1)
        v_len = v.size(1)

        # 得到全局 QKV
        Q = self.q_linear_trans(q)
        K = self.k_linear_trans(k)
        V = self.v_linear_trans(v)

        # 拆分多头
        # 注意： 为啥不是一开始初始化的时候就进行拆分， 因为工程上的妥协， 初始拆分的话， 要初始化 28个 linear
        # 原始 shape [batch_size, sentence_len, d_model] => [batch_size, sentence_len, self.num_head, self.d_k]
        # 也就是将 d_model 拆分为 num_head， d_k
        # [2,4,512]  => [2,4,8,64]  => [2,8,4,64]
        # transpose 交换维度的位置， 因为我们想要的是 [8个批次，句子， 64维度]
        Q = Q.view(batch_size, sentence_len, self.num_head, self.d_k).transpose(1,2)
        K = K.view(batch_size, k_len, self.num_head, self.d_k).transpose(1,2)
        V = V.view(batch_size, v_len, self.num_head, self.d_k).transpose(1,2)

        # 执行注意力计算, 缩放、归一化, 得到新向量
        context = self.handler_attention(Q, K, V, mask)

        # 拼接回原来的矩阵 [2,8,4,64] => [2, 4, 8, 64] => [2, 4, 512]
        context = context.transpose(1, 2).contiguous().view(batch_size, sentence_len, self.d_model)

        #  过一遍线性层， 混合所有头的信息, 其实这里给了最后一次调整的机会
        # [2, 4, 512] @ [512, 512] 看成 [1, 512, 512] => [2, 4, 512]
        last_result = self.o_linear_trans(context)

        # 这是经过了注意力机制字后的最终矩阵
        return last_result



    def handler_attention(self, Q, K, V, mask = None):
        """
        执行注意力计算
        """
        # 核心数学公式：Attention(Q, K, V) = softmax(Q * K ^ T / sqrt(d_k)) * V

        # 1 计算Q * K ^ T / sqrt(d_k)  matmul 矩阵乘法， 仅仅只对最后两个维度进行操作
        # Q K V [2,8,4,64]
        # score [2,8,4,64] @ [2,8,64,4] => [2,8,4,4]
        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt( self.d_k )

        # 处理遮罩 mask
        if mask is not None:
            # 将为 True 的地方， 填充极小的 -1e9
            score = score.masked_fill(mask, -1e9)

        # 2 计算 softmax 归一化，将分数换成 0-1 的权重， 这是真正的注意力权重矩阵
        # dim 指的是对最后一个维度进行 softmax 运算
        softmax_result = F.softmax(score, dim=-1)

        # 3 计算 * V 根据权重矩阵 加权得到最终的注意力矩阵   [2,8,4,4] @ [2,8,4,64] => [2,8,4,64]
        context = torch.matmul(softmax_result, V)

        return context
