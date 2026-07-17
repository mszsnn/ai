# 翻译句子
import torch
import torch.nn.functional as F

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
    src_tensor = torch.tensor([src_ids], dtype= torch.long, device=device)

    # Encoder 只依赖源句子，推理时提前算一次即可，后续每一步复用
    with torch.no_grad():
        src_mask = model.make_encoder_mask(src_tensor)
        encoder_output = model.encode(src_tensor, src_mask)

    # 一开始，我们只给 Decoder 一个空荡荡的开始符 <SOS>
    decoder_input_ids = [tgt_tokenizer.SOS_ID]


    # 开始循环输出

    for step in range(max_len):
        decoder_input_tensor = torch.tensor([decoder_input_ids], dtype=torch.long, device=device)

        # 前向传播, 不要计算梯度
        with torch.no_grad():
            target_mask = model.make_decoder_mask(decoder_input_tensor)
            logits = model.decode(
                decoder_input_tensor,
                encoder_output,
                target_mask,
                src_mask
            )

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


def beam_search_translate_sentence(model, sentence, src_tokenizer, tgt_tokenizer, max_len, device, beam_width = 3):
    """
        使用集束搜索 (Beam Search) 实现自回归解码。
        像真正的大模型一样，一个词一个词地往外蹦！
        """

    # 极其重要：切换到评估模式。关闭 Dropout 等随机机制，保证每次翻译结果稳定
    model.eval()

    # 原始 ids
    src_ids = src_tokenizer.encode(sentence, max_len)

    # 增加 batch 维度  [sentence_len] => [1, sentence_len]
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    # Encoder 只依赖源句子，beam 的所有分支共享同一份编码结果
    with torch.no_grad():
        src_mask = model.make_encoder_mask(src_tensor)
        encoder_output = model.encode(src_tensor, src_mask)

    # 初始化beam容器， 容器格式 [ (id 列表, 得分) ]

    beams = [([tgt_tokenizer.SOS_ID], 0.0)]

    # 存储遇到了 EOS 结束符的完整句子方案
    completed_beams = []

    for step in range(max_len):
        new_beams = []

        for seq, score in beams:
            # 里面和贪心算法一样
            decoder_input_tensor = torch.tensor([seq], dtype=torch.long, device=device)
            # 前向传播, 不要计算梯度
            with torch.no_grad():
                target_mask = model.make_decoder_mask(decoder_input_tensor)
                logits = model.decode(
                    decoder_input_tensor,
                    encoder_output,
                    target_mask,
                    src_mask
                )
            # 得到最后一步生成的词的 logits
            last_logits = logits[0, -1, :]

            # 🔥 架构师级防御编程：Logits Masking
            # 强制干掉 PAD 和 UNK，让它们的概率彻底变成 0
            last_logits[tgt_tokenizer.PAD_ID] = -float('Inf')
            last_logits[tgt_tokenizer.UNK_ID] = -float('Inf')

            #本来这里所有的概率应该是相乘， 但是相乘容易溢出， log(A X B) = log(A) + log(B) 所以改成 相加

            log_probs = F.log_softmax(last_logits, dim = -1)

            #从这个list里面获取 前 beam_width 个
            top_log_probs, top_ids = torch.topk(log_probs, beam_width)

            # 裂变宇宙
            for i in range(beam_width):
                next_token_id = top_ids[i].item()
                next_score = score +  top_log_probs[i].item()
                new_beams.append((seq + [next_token_id], next_score))

        # beams 循环结束之后， 可能产生 beams_len * beams_width 个可能性，从这里面取出来 top beams_width个
        # ==================================================
        # 4. 优胜劣汰与长度惩罚 (Length Penalty)
        # ==================================================
        # 句子越长，累加的负数分数越多。为了公平，我们将分数除以 (句子长度的 0.7 次方) 进行平衡
        # 根据这个平衡后的分数进行降序排列，只保留前 beam_width 个最强者
        new_beams = sorted(new_beams, key = lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)[:beam_width]

        # 筛选：哪些宇宙已经走到了终点 (<EOS>)，哪些还需要继续发展
        beams = []
        for seq, score in new_beams:
            if seq[-1] == tgt_tokenizer.EOS_ID:
                completed_beams.append((seq, score))
            else:
                beams.append((seq, score))

        # 如果所有分支都结束了， 停止推理
        if len(beams) == 0:
            break


    # 最终会有 completed_beams  和 beams 个来决定最终翻译结果
    all_final_beams = completed_beams + beams

    if not all_final_beams:
        return ""

    # 用长度惩罚后的得分选出终极冠军 🏆
    best_beam = max(all_final_beams, key=lambda x: x[1] / (len(x[0]) ** 0.7))
    best_seq = best_beam[0]
    return tgt_tokenizer.decode(best_seq)
