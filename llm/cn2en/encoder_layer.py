import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from feedforword_neural_network import  FeedforwardLayer

# Encoder block 终极拼接

class EncoderBlock(nn.Module):
    """
    一个完整的 Encoder Block
    大模型就是把这个Block 重复堆叠N次， （比如 GPT-3 堆了 96 层）。

    注意力机制
    X => Attention => ADD 残差链接 => LayerNorm 层归一化

    前馈神经网络
    注意力机制的结果 => ADD 残差链接  => LayerNorm 层归一化

    残差链接用来 帮助深层网络梯度更新向前
    层归一化，将 512 的坐标整体进行进行整理

    """
    def __init__(self, d_model, num_head, high_dimension, dropout = 0.1):
        super().__init__()

        # 初始化
        self.attention_result = MultiHeadAttention(d_model, num_head)
        self.feedforward_layer = FeedforwardLayer(d_model, high_dimension, dropout)

        # 初始化层归一
        self.layer_norm_attention = nn.LayerNorm(d_model)
        self.layer_norm_feedforward = nn.LayerNorm(d_model)

        # 防止过拟合
        self.dropout = nn.Dropout(dropout)



    def forward(self, x, mask = None):
        """
        注意力机制
        X => Attention => ADD 残差链接 => LayerNorm 层归一化

        前馈神经网络
        注意力机制的结果 => ADD 残差链接  => LayerNorm 层归一化
        """

        # 保留原始

        old = x
        # 执行 forward
        x = self.attention_result(x, x, x, mask)
        #  防止过拟合
        x = self.dropout(x)
        # 残差相加， 之后进行归一化
        x = self.layer_norm_attention( old + x )


        old = x
        x = self.feedforward_layer(x)
        x = self.dropout(x)
        x = self.layer_norm_feedforward(old + x)  # 先加，后归一化

        return x










