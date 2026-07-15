# 翻译句子
import torch

def translate_sentence(model, sentence, src_tokenizer, tgt_tokenizer, max_len, device):
    """
    使用贪心搜索 (Greedy Search) 实现自回归解码。
    像真正的大模型一样，一个词一个词地往外蹦！
    """

    # 极其重要：切换到评估模式。关闭 Dropout 等随机机制，保证每次翻译结果稳定
    model.eval()

    # 原始 ids
    src_ids = src_tokenizer.encode(sentence, max_len)

    # 增加 batch 维度  [sentence_len] => [1, sentence_len]
    src_tensor = torch.tensor([src_ids], dtype= torch.long)

    # 一开始，我们只给 Decoder 一个空荡荡的开始符 <SOS>
    decoder_input_ids = [tgt_tokenizer.SOS_ID]


    # 开始循环输出

    for step in range(max_len):
        decoder_input_tensor = torch.tensor([decoder_input_ids], dtype=torch.long)

        # 前向传播, 不要计算梯度
        with torch.no_grad():
            logits = model(src_tensor, decoder_input_tensor)

        # 用贪心搜索，每一步都找最大的概率
        # # logits[0] 取出当前 batch，[-1] 取出序列最后一步，[:] 取出整个词表的打分
        lats_token_logits = logits[0, -1, :]

        # 找到最大的
        predicted_id = torch.argmax(lats_token_logits).item()

        if predicted_id == tgt_tokenizer.EOS_ID:
            break

        decoder_input_ids.append(predicted_id)

    # 解码
    translate_text = tgt_tokenizer.decode(decoder_input_ids[1:])

    return translate_text



