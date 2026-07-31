# 基础表示层，目的是将上一步输出的  ids 的纯数字序列，赋予真正的意义
import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    将离散的id序列，转化为高维度的向量
    """
    def __init__(self, vocab_size, d_model):
        """
        维度
        """
        # 继承 nn.Module  调用 父类初始化函数
        super().__init__()

        # 核心操作， 构建词向量矩阵，Embedding 用来提供矩阵的索引
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model



    def forward(self, x):
        """
        父类中会调用这个函数， 相当于是子类必须要实现的方法， 作用是 作为一个神经网络模块 的输出. 不用手动调， 内部会自己调用

       input x shape [batch_size, sentence_word_length]
       output shape [batch_size, sentence_word_length, d_model]

        例如：
        shape: [2,4] 2条句子， 每个句子 4个id 长
        [
            [1,2,3,4],
            [2,3,4,5]
        ]

        返回
        [
            [
                [-1.234, 0.56, ....d_model]
                [-9.234, 0.12, ....d_model]
                [-1.234, 0.34, ....d_model]
                [-0.34, 0.78, ....d_model]
            ],
            [
                []
                []
                []
                []
            ]
        ]


        """
        return  self.embedding(x) * math.sqrt(self.d_model)



class PositionEncoding(nn.Module):
    """
    位置编码层：给没有任何顺序概念的词向量，注入位置信息。
    """
    def __init__(self,  d_model, max_len = 5000):
        super().__init__()

        # 生成全 0矩阵 维度 【max_len, d_model】
        # [
        #     [0,0,0,0,0,0,....0],
        #     [0,0,0,0,0,0,....0],
        #     .....,
        #      max_len 行 [0,0,0,0,0,0,....0]
        # ]


        pe = torch.zeros(max_len, d_model)

        # 生成 绝对位置列向量
        # [
        #     [0],
        #     [1],
        #     [2],
        #     [3],
        #     [4],
        #     ...
        #     [max_len],
        # ]

        # torch.arange(0, max_len, dtype=torch.float) 的结果是 【max_len】
        # unsqueeze(1) 在 1 的位置新增一个维度  [max_len]  -> [max_len, 1]
        # unsqueeze(0) 在 0 的位置新增一个维度  [max_len]  -> [1, max_len]

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算 除数
        # 公式： 1 / ( 10000 ^ (2*i / d_modal)) 肯定不能直接计算， 会溢出

        # torch.arange(0, d_model, 2) 是间隔  2 取一次

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        # 0::2 从 0 开始， 步长 为 2 获取 列 0 2 4 6 ... 偶数列
        # position * div_term  [max_len, 1] * [d_model / 2] 会被看成 [max_len, 1] * [1, d_model / 2] => [max_len, d_model / 2]

        # 对应位置赋值
        pe[:, 0::2] = torch.sin( position * div_term)

        pe[:, 1::2] = torch.cos( position * div_term)

        # 增加 batch 维度变成 【1，max_len, d_model】方便后面进行相加
        pe = pe.unsqueeze(0)

        # 注册为 Buffer
        # Buffer 意思是模型的一部分， 后续保存加上，他是数学常量

        self.register_buffer('pe', pe)



    def forward(self, x):
        # x 维度 【batch_size, sentence_len, d_model】

        # pe 维度 【1， max_len, d_model】
        # x.size(1) 代表取 x 维度 1 位置的大小 => sentence_len, 因为 有几个就读取几行

        # 会进行广播， 每个句子都会 加上位置信息

        return x + self.pe[:, :x.size(1), :]


if __name__ == "__main__":
    # 假设 Tokenizer 字典大小是 10000，模型维度设定为 512
    vocab_size = 10000
    d_model = 512

    # 模拟一个 Batch: 2句话，每句话 10 个词
    # 形状 [Batch_size, Seq_Len]
    mock_ids = torch.tensor([
        [2, 56, 12, 3, 0, 0, 0, 0, 0, 0],
        [2, 89, 45, 11, 23, 3, 0, 0, 0, 0]
    ])
    print(f"1. Tokenizer 输出的 ID 序列形状: {mock_ids.shape}")

    # 实例化我们的两个基座
    emb_layer = TokenEmbedding(vocab_size, d_model)
    pos_layer = PositionEncoding(d_model)

    # 前向传播测试
    x = emb_layer(mock_ids)
    print(f"2. 经过 Embedding 层后的张量形状: {x.shape} -> (已获得语义)")

    x = pos_layer(x)
    print(f"3. 经过 Positional Encoding 后的形状: {x.shape} -> (已注入位置)")
