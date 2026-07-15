import json
import random

import torch
from torch.utils.data import Dataset
from tokenizer import  BasicTokenizer



def load_jsonl(file_path):
    """
    读取 JSONL 格式的中英翻译数据。
    文件中的每一行格式：
        {"zh": "我 爱 你", "en": "i love you"}
    返回格式：
        [
            {"zh": "我 爱 你", "en": "i love you"},
            ...
        ]
    """
    data_list = []
    with open(file_path, 'r', encoding = 'utf8') as file:
        for line in file:
            line = line.strip()
            # 将当前一行 JSON 字符串转成 Python 字典
            item = json.loads(line)

            en = item['en'].strip()
            zh = item['zh'].strip()

            if en and zh:
                data_list.append({
                    'en': en,
                    'zh': zh
                })
    return data_list




def  split_raw_data(
    data_list,
    train_ratio = 0.8,
    val_ratio = 0.1,
    seed = 42
):
    """
        将原始文本数据划分为：
            训练集
            验证集
            测试集
        默认比例：
            训练集：80%
            验证集：10%
            测试集：10%
        为什么在创建 Dataset 前划分？
            因为中英文词表只能使用训练集构建。
            正确顺序：
                原始数据
                ↓
                划分 train / val / test
                ↓
                只使用 train 构建词表
                ↓
                创建三个 TranslationDataset
        seed 用来保证每次运行得到相同的数据划分。
        """
    origin_data = data_list.copy()
    total_size = len(origin_data)

    # 创建固定随机种子生成器
    random_generator = random.Random(seed)
    # 随机打乱
    random_generator.shuffle(origin_data)

    # 计算训练集和验证集数量
    train_size = int( train_ratio * total_size )
    val_size = int( val_ratio * total_size)

    train_data = origin_data[:train_size]

    val_end = train_size + val_size
    val_data = origin_data[train_size:val_end]
    test_data = origin_data[val_end:]

    return train_data, val_data, test_data




class TranslationDataset(Dataset):
    """
        中译英 Transformer 数据集。
        每条原始数据：
            {
                "zh": "我 爱 你",
                "en": "i love you"
            }
        最终返回三个张量：
            src
            decoder_input
            labels
        假设英文句子：
            i love you
        对应目标 Token：
            [i, love, you]
        Teacher Forcing 构造结果：
            decoder_input：
                [SOS, i, love, you]
            labels：
                [i, love, you, EOS]
        对齐关系：
            Decoder 输入       正确答案
            SOS               i
            i                 love
            love              you
            you               EOS
        """
    def __init__(self, data_list, src_tokenizer, tgt_tokenizer, max_src_len = 64, max_tgt_len = 64 ):
        super().__init__()

        # 保留原始文本
        self.data_list = data_list

        # 中文和英文分别使用自己的 Tokenizer
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        # 中英文使用自己的最大长度
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len


    def __len__(self):
        """
        dataloader 会通过这个方法获取 Dataset 长度
        """
        return len(self.data_list)


    def __getitem__(self, index):
        """
        根据 index 获取一条数据
        返回
        src_tensor
        decoder_input_tensor
        labels_tensor
        """
        # ====================================
        # 1. 获取原始中英文文本
        # ====================================
        item = self.data_list[index]

        src_text = item['zh']
        tgt_text = item['en']

        # 给 encoder 的 SOS ids EOS PAD PAD
        src_ids = self.src_tokenizer.encode(src_text, self.max_src_len)

        # 给 decoder 的 decoder_input labels
        decoder_input, labels = self.tgt_tokenizer.encode_target(tgt_text, self.max_tgt_len)

        # Embedding 层要求输入:Long Tensor  所以需要转换为 torch.Tensor

        src_tensor = torch.tensor(src_ids, dtype=torch.long)
        decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return src_tensor, decoder_input_tensor, labels_tensor



if __name__ == '__main__':

    data = load_jsonl('translation_dataset.jsonl')
    train_data, val_data, test_data = split_raw_data(data)

    # 源
    src_tokenizer = BasicTokenizer()
    tgt_tokenizer = BasicTokenizer()

    src_sentences = [
        item['zh']
        for item in train_data
    ]

    tgt_sentences = [
        item['en']
        for item in train_data
    ]

    src_tokenizer.build_word_sheet(
        src_sentences,
        min_frequency=1
    )

    tgt_tokenizer.build_word_sheet(
        tgt_sentences,
        min_frequency=1
    )

    print(
        "src vocab:",
        src_tokenizer.vocab_size
    )

    print(
        "tgt vocab:",
        tgt_tokenizer.vocab_size
    )

    train_dataset = TranslationDataset(
        data_list=train_data,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_src_len=64,
        max_tgt_len=64
    )

    val_dataset = TranslationDataset(
        data_list=val_data,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_src_len=64,
        max_tgt_len=64
    )

    test_dataset = TranslationDataset(
        data_list=test_data,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_src_len=64,
        max_tgt_len=64
    )

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    src, decoder_input, labels = train_dataset[0]

    print("====================")
    print("单条数据测试")
    print("====================")

    print(
        "src:",
        src.shape
    )

    print(
        "decoder_input:",
        decoder_input.shape
    )

    print(
        "labels:",
        labels.shape
    )

    # ====================================
    # 8. 测试 DataLoader batch
    # ====================================

    src, decoder_input, labels = next(
        iter(train_loader)
    )

    print("====================")
    print("Batch测试")
    print("====================")

    print(
        "src:",
        src.shape
    )

    print(
        "decoder_input:",
        decoder_input.shape
    )

    print(
        "labels:",
        labels.shape
    )








