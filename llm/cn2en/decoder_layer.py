import torch.nn as nn
from torch.nn.functional import dropout

from multi_head_attention import MultiHeadAttention
from feedforword_neural_network import  FeedforwardLayer


# Decode layer


class DecoderBlock(nn.Module):
    """
        一个完整的 Decoder Block
        包含三个子层：
        1. 掩码自注意力层 (Masked Self-Attention)
        2. 交叉注意力层 (Cross-Attention)
        3. 前馈神经网络 (FeedForward)
    """

    def __init__(self, d_model, num_head, high_dimension, dropout = 0):
        super().__init__()

        # 1. 实例化两个独立的注意力机制
        # 注意：这里我们复用了同一个 MultiHeadAttention 类！

        # 自注意力机制
        self.self_attention = MultiHeadAttention( d_model, num_head)

        # 交叉注意力机制
        self.cross_attention = MultiHeadAttention(d_model, num_head)

        # 前馈神经网络
        self.feedforward_layer = FeedforwardLayer(d_model, high_dimension, dropout)


        # 防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 三层归一
        self.layer_norm_attention_self = nn.LayerNorm(d_model)
        self.layer_norm_attention_cross = nn.LayerNorm(d_model)
        self.layer_norm_feedforward = nn.LayerNorm(d_model)



    def forward(self, x, encoder_output, look_mask = None, padding_mask = None):
        """
        :param x: 解码器当前的输入序列张量，shape [batch_size, tgt_len, d_model]
        :param encoder_output: 编码器最终输出的原文特征张量，shape [batch_size, src_len, d_model]
        :param look_mask: 下三角掩码，防止解码器偷看未来的词
        :param padding_mask: （可选）用于遮蔽 Encoder 输出中的 <PAD> 占位符
        """

        # ==========================================
        # 子层 1：掩码自注意力机制 (Masked Self-Attention)
        # 目标：让解码器理解自己当前已经生成的内容，但不看未来
        # ==========================================

        old = x
        x = self.self_attention(x, x, x, mask = look_mask)
        x = self.dropout(x)
        x = self.layer_norm_attention_self(old + x)

        # ==========================================
        # 子层 2：交叉注意力机制 (Cross-Attention)
        # 目标：拿着解码器当前的逻辑(Q)，去原文特征(K,V)里提取需要的信息
        # ==========================================
        old = x
        x = self.cross_attention(x, encoder_output, encoder_output, mask=padding_mask)
        x = self.dropout(x)
        x = self.layer_norm_attention_cross(old + x)

        # ==========================================
        # 子层 3：前馈神经网络 (FeedForward Network)
        # 目标：对混合了原文和译文信息的特征进行非线性深度加工
        # ==========================================

        old = x
        x = self.feedforward_layer(x)
        x = self.dropout(x)
        x = self.layer_norm_feedforward(old + x)

        return x





