# 将 句子 转化为 ids， 注意这里面还存在分词的处理
import json


# 注意： 中英文是分来编码的

class BasicTokenizer:
    """
    简易版词表和分词器，（基于空格分隔的语料）
    目标是将自然语言文本映射到模型可计算的证书 id 序列
    """

    def __init__(self):

        self.PAD_ID = 0  # Padding 用来填充句子长度 ， 模型不应该关注他
        self.UNK_ID = 1  # Unknown 遇到词表中没有的生僻词的时候，统一用它代替
        self.SOS_ID = 2  # Start Of Sentence 代表句子的开始
        self.EOS_ID = 3  # End Of Sentence 代表句子结尾

        # 定义核心字典映射初始化

        self.word2id = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }

        self.id2word = {
            0: '<PAD>',
            1: '<UNK>',
            2: '<SOS>',
            3: '<EOS>'
        }



    def build_word_sheet(self, sentences, min_frequency = 1):
        """
        构建词汇表 (Vocabulary)
        :param sentences: 字符串列表，例如 ["我 爱 深度 学习", "深度 学习 改变 世界"]
        :param min_frequency: 词频阈值，低于此频率的词会被丢弃（变成 UNK），有效控制词表爆炸

        先统计词频率， 然后高频词加入字典映射， 分配 ID
        低频次不出现在 字典中
        """
        # 统计词频
        words_count = {}
        for sentence in sentences:
            for word in sentence.split():
                words_count[word] = words_count.get(word, 0) + 1

        # 添加 ID


        for key, value in words_count.items():
            if value >=  min_frequency and key not in self.word2id:
                l = len(self.id2word)
                self.word2id[key] = l
                self.id2word[l] = key



    def save_vocab(self, file_name= 'vocab.json'):
        with(
            open(file_name, 'w', encoding='utf-8') as f
        ):
            f.write(json.dumps(self.word2id, ensure_ascii=False))

        print(f'词表已经写入到{file_name}')


    def encode_tokens(self, sentence):
        """
        文本 -> 原始token id
        不添加任何特殊符号
        """
        ids = [self.word2id.get(item, self.UNK_ID) for item in sentence.split()]
        return ids


    def encode(self, sentence, max_len):
        """
        核心编码器：将一句话转化为等长的 ID 张量基础数据
        :param sentence: 输入的字符串，如 "我 爱 学习"
        :param max_len: 序列的最大长度，这是为了保证 GPU 可以进行批量(Batch)矩阵运算
        """

        ids = self.encode_tokens(sentence)

        # 首尾添加开始和结束 list 可以直接用 + 号， 但是只能list + list
        ids = [self.SOS_ID] + ids + [self.EOS_ID]

        # 拼接长度或者截取
        count = len(ids)
        if count > max_len:
            # 截取， 必须保证最后一个必须是 EOS
            ids = ids[:max_len - 1] + [self.EOS_ID]
        else :
            # 补
            ids = ids + [self.PAD_ID] * (max_len - count)

        return ids

    def encode_target(self,sentence, max_len):
        """
        Transformer Decoder 训练专用
         输入:
            I love you
        decoder_input:
            [
                SOS,
                I,
                love,
                you
            ]
        labels:
            [
                I,
                love,
                you,
                EOS
            ]
        两者错开一个位置。

        这是 Teacher Forcing 的核心。
        """

        ids = self.encode_tokens(sentence)

        # 最大长度 max_len, 空出来一个位置 添加 SOS / EOS
        ids = ids[:max_len-1]

        # teacher forcing 输入
        decoder_input = [self.SOS_ID] + ids

        # labels  loss 计算的核心
        labels = ids + [self.EOS_ID]

        # 处理长度
        decoder_input = decoder_input + [self.PAD_ID] * (max_len - len(decoder_input))

        labels = labels + [self.PAD_ID] * (max_len - len(labels))

        return decoder_input, labels



    def decode(self, ids):
        """
        核心解码器：将模型输出的 ID 序列还原成人类可读的文字
        :param ids: 整数列表，如 [2, 18, 4, 3, 0, 0]
        """

        sentence = []

        for item in ids:
            if item == self.EOS_ID:
                break
            if item == self.UNK_ID or item == self.SOS_ID:
                continue
            sentence.append(self.id2word.get(item, '<UNK>'))
        return ' '.join(sentence)

    @property
    def vocab_size(self):
        return len(self.word2id)




if __name__ == '__main__':
    # 1. 实例化
    tokenizer = BasicTokenizer()

    # 2. 模拟你读取到的真实分词数据
    mock_corpus = [
        "1998年 , 经过 统一 部署 , 伊犁州 , 地 两 级 党委 开始",
        "群众 在 草地 , 球场 , 树 荫 下 席地而坐 "
    ]

    # 3. 建立词典
    tokenizer.build_word_sheet(mock_corpus, min_frequency=1)

    tokenizer.save_vocab()
    print(f"✅ 词表构建完成，大小为: {tokenizer.vocab_size}")

    # 4. 测试编码 (输入到模型前)
    test_sentence = "1998年 , 群众 在 草地 休息"
    encoded_ids = tokenizer.encode(test_sentence, max_len=20)
    print(f"🔢 编码后的 ID 序列: {encoded_ids}")

    # 5. 测试解码 (模型输出后还原)
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"📝 解码还原文本: {decoded_text}")


