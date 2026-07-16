# 掩码的形状: torch.Size([1, 1, 4, 4])
# 掩码的内容:
#  tensor([[1., 0., 0., 0.],
#          [1., 1., 0., 0.],
#          [1., 1., 1., 0.],
#          [1., 1., 1., 1.]])
import torch


def generate_mask(sentence_len, device=None):
    """
    定义 True 为最终要覆盖的位置
    生成用于 Decoder 自回归训练的掩码 (Look-ahead Mask)
    :param sentence_len: 目标序列的长度 (Target Sequence Length)
    :param device
    :return: shape 为 [1, 1, seq_len, seq_len] 的掩码张量

    """

    # 第一步：生成一个 seq_len x seq_len 的全 1 正方形矩阵
    # 比如 seq_len=4，就是 4x4 的全 1 矩阵

    mask = torch.ones(sentence_len, sentence_len, dtype=torch.bool, device = device)

    # 对于主对角线位置机器位置以下的值保留，
    # diagonal=0 位置 就是  [0, 0] [1, 1] [2,2]...
    # diagonal=1 位置 就是  [0, 1] [1, 2] [2,3]...
    # diagonal=-1 位置 就是  [1, 0] [2, 1]  [3, 2]...

    # tril 保留下半部分。 triu  保留上半部分


    mask = torch.triu(mask, diagonal=1)

    # 第三步：为了能和多头注意力的 Score 矩阵 [Batch, Heads, Seq_Len, Seq_Len] 相加/遮蔽
    # 我们需要在前面扩展出 Batch 和 Head 的维度，利用 PyTorch 的广播机制 (Broadcasting)

    # 在 0 的位置增加维度
    mask = mask.unsqueeze(0).unsqueeze(0)

    return mask


# --- 测试代码 ---
if __name__ == "__main__":
    # 假设我们现在要翻译的英文目标句子长度是 4 ("<SOS>", "I", "love", "you")
    test_len = 4
    mask = generate_mask(test_len)

    print("掩码的形状:", mask.shape)
    print("掩码的内容:\\n", mask[0][0])




