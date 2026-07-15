# 最终组装为编码器
import torch.nn as nn

from represent_layer import TokenEmbedding, PositionEncoding
from encoder_layer import  EncoderBlock

class Encoder(nn.Module):
    """
    Encoder 宏观大楼：
    负责将离散的 Token ID 序列，转化为富含上下文语义和位置信息的连续高维张量。
    """

    def __init__(self, vocab_size, d_model, num_head, high_dimension, num_layers, max_len= 5000, dropout = 0):
        """
          参数对齐说明：
          - vocab_size: 词表大小 (由 Tokenizer 决定)
          - d_model: 模型维度 (例如 512)
          - num_head: 多头注意力的头数 (例如 8)
          - high_dimension: FFN 的高维空间维度 (例如 2048)
          - num_layers: 堆叠多少层 EncoderBlock (例如 6)
          - max_len 位置矩阵的最大长度
          - dropout 失活概率
          """

        super().__init__()

        self.d_model = d_model
        # 实例化词嵌入矩阵
        self.embedding = TokenEmbedding(vocab_size, d_model)

        # 实例化位置信息
        self.position = PositionEncoding(d_model, max_len)

        # 根据层数， 创建 layer 实例, 创建 num_layers 个 堆叠层
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_head, high_dimension, dropout)
            for _ in range(num_layers)
        ])

        # 这里多加一层 归一化处理， 因为 2017 年论文刚出来， 和目前的 EncoderBlock 的归一化处理有一些区别，
        # 两种方式 Post-LN vs Pre-LN  如果是前者， 不用多加 归一化， 如果是后者， 需要多加归一化，因为后者是提前处理的参数， 再残差链接

        self.layer_norm = nn.LayerNorm(d_model)

        # 随机失活
        self.dropout = nn.Dropout(dropout)



    def forward(self, x, mask=None):
        # x 形状 [batch_size, sentence_len]

        #===============第一阶段基础表示=========
        # 构建词向量矩阵, 通过索引， 得到 词向量
        x = self.embedding(x)

        # 添加上位置信息
        x = self.position(x)

        # 随机失活
        x = self.dropout(x)

        # ===============第二阶段 深层特征提取 =========

        for layer in self.layers:
            x = layer(x, mask)

        # 最终进行一次归一化

        x = self.layer_norm(x)

        return x














