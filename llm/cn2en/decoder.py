# 最终组装为解码器
import torch.nn as nn

from represent_layer import TokenEmbedding, PositionEncoding
from decoder_layer import  DecoderBlock



class Decoder(nn.Module):
    """
    Decoder 宏观大楼：
    负责将目标序列（预测文本）ID 转换为高维稠密特征，
    结合 Encoder 的记忆矩阵进行多层加工，最终输出全词表的概率分布。
    """

    def __init__(self, vocab_size, d_model, num_head, high_dimension, num_layers, max_len= 5000, dropout = 0):
        """
          参数对齐说明：
          - vocab_size: 词表大小 (由 Tokenizer 决定)
          - d_model: 模型维度 (例如 512)
          - num_head: 多头注意力的头数 (例如 8)
          - high_dimension: FFN 的高维空间维度 (例如 2048)
          - num_layers: 堆叠多少层 DecoderBlock (例如 6)
          - max_len 位置矩阵的最大长度
          - dropout 失活概率
          """
        super().__init__()

        self.d_model = d_model

        self.embedding = TokenEmbedding(vocab_size, d_model)

        self.position = PositionEncoding(d_model, max_len = max_len)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_head, high_dimension, dropout)
            for _ in range(num_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(d_model)

        # 随机失活
        self.dropout = nn.Dropout(dropout)

        # 构造目标答案
        # 把 d_model (如512) 维映射成全词表维 (Vocab_Size)，用于计算交叉熵损失或挑出最高概率词
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, target, encoder_output, look_mask = None, padding_mask = None ):
        """
        :param target: 目标词序列的 Token ID, shape: [batch_size, tgt_len]
        :param encoder_output: Encoder 层的最终输出特征, shape: [batch_size, src_len, d_model]
        :param look_mask: 下三角掩码，遮蔽未来
        :param padding_mask: 遮蔽原文 padding
        :return: 词表预测对数几率 (Logits), shape: [batch_size, tgt_len, vocab_size]
        """

         #===============第一阶段基础表示=========
        # 构建词向量矩阵, 通过索引， 得到 词向量
        x = self.embedding(target)

        # 添加上位置信息
        x = self.position(x)

        # 随机失活
        x = self.dropout(x)

        # --- 2. 穿越 N 层 Decoder 迷宫 ---
        # 每一层都需要重复输入相同的 encoder_output 和对应的掩码
        for layer in self.layers:
            x = layer(
                x=x,
                encoder_output=encoder_output,
                look_mask=look_mask,
                padding_mask=padding_mask
            )

        x = self.final_layer_norm(x)

        last_output = self.output_linear(x)

        return last_output




