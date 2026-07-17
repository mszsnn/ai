import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from mask_util import  generate_mask

class Transformer(nn.Module):
    """
    Transformer 总模型。
    主要职责：
    1. 创建 Encoder Padding Mask
    2. 创建 Decoder Padding Mask + Look-ahead Mask
    3. 调用 Encoder 编码源语言
    4. 调用 Decoder 根据目标语言前缀和 Encoder 输出进行预测
    """

    def __init__(self, src_vocab_size,target_vocab_size, d_model, num_head, high_dimension, num_layers, pad_id = 0, max_len= 5000, dropout = 0):
        super().__init__()

        # 编码器， 输出上下文词义和位置关系向量
        self.encoder = Encoder(
            src_vocab_size, d_model, num_head, high_dimension, num_layers, max_len, dropout
        )

        # 解码器， 并且得到最终的原始打分
        self.decoder = Decoder(
            target_vocab_size, d_model, num_head, high_dimension, num_layers, max_len, dropout
        )

        self.pad_id = pad_id


    def make_encoder_mask(self, src):
        """
        生成 Encoder 的掩码：只遮蔽 <PAD> 占位符
        src 形状: [batch_size, src_len]
        需要的形状 [batch_size, num_head ,query_len, key_len]  QK^T 之后的形状 query_len, key_len，
        """
        # src 形状 [batch_size, sentence_len]
        # 将 src 是经过词表后， 词表中的 id 序列

        # [
        #     [1，2, 4, 6, 0, 0, 0, 0],
        #     [1，2, 4, 6, 0, 0, 0, 0]
        # ]
        #         =>
        # [
        #     [False，False, False, False, Ture, Ture, Ture, Ture],
        #     [False，False, False, False, Ture, Ture, Ture, Ture]
        # ]

        # 但是还需要 增加维度

        # 经 token 为 pad 的地方， 填充为 -1e9因为这些值不应该得到编码器的关注

        src_mask = (src == self.pad_id).unsqueeze(1).unsqueeze(2)

        return src_mask




    def make_decoder_mask(self, target,):
        sentence_len = target.size(1)
        #  target [batch_size, tgt_len]

        target_padding_mask = (target == self.pad_id).unsqueeze(1).unsqueeze(2)

        target_look_mask = generate_mask(sentence_len, target.device)

        # pad | 未来未知都需要屏蔽
        # | & 本来是按位操作符， 但是两边是 Torch 张量， 重载了 | 变成了 按位置求或  且操作
        end_mask = target_look_mask | target_padding_mask

        return end_mask


    def encode(self, src, src_mask):
        encoder_output = self.encoder(src, mask=src_mask)
        return encoder_output


    def decode(self, target, encoder_output, target_mask, src_mask):
        decoder_output = self.decoder(
            target,
            encoder_output,
            look_mask=target_mask,
            padding_mask=src_mask
        )

        return decoder_output


    def forward(self, src, target):
        """
        Transformer 前向传播。

        Parameters
        ----------
        src:
            源语言 Token IDs。

            shape:
                [batch_size, src_len]

        target:
            Decoder 输入 Token IDs。

            shape:
                [batch_size, tgt_len]

        Returns
        -------
        logits:
            对目标词表中每一个 Token 的原始预测分数

            shape:
                [batch_size, tgt_len, tgt_vocab_size]
        """

        # 1 创建 mask
        src_mask = self.make_encoder_mask(src)
        target_mask = self.make_decoder_mask(target)

        # 2 encoder

        encoder_output = self.encode(src, src_mask)

        # 3 decoder
        # 这里有一个很重要的细节： 为啥 padding_mask 用的是 src_mask?
        # padding_mask 传递够了交叉注意力层， 交叉注意力中的 KV 来自 encoder ,
        # padding_mask是 QK^T 后得到了打分矩阵，每一行代表了 Q Token 对所有 K Token 的打分，
        # padding_mask 其实就是阻止 K 中的 pad 被 Q关注到

        decoder_output = self.decode(
            target,
            encoder_output,
            target_mask,
            src_mask
        )


        return decoder_output

