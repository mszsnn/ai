import os.path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from llm.cn2en.checkpoint import save_checkpoint, load_checkpoint
from llm.cn2en.tokenizer import BasicTokenizer

from llm.cn2en.translation_dataset import (
    load_jsonl,
    split_raw_data,
    TranslationDataset
)

from llm.cn2en.transformer import Transformer

from epoch_core import  train_one_epoch
from translate_sentence import translate_sentence, beam_search_translate_sentence

# ==================================================
# 1. 设备
# ==================================================

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")




# ==================================================
# 2. 读取数据
# ==================================================

data = load_jsonl(
    "translation_dataset.jsonl"
)


# ==================================================
# 3. 划分 train / val / test
# ==================================================

train_data, val_data, test_data = split_raw_data(
    data,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42
)

src_tokenizer = BasicTokenizer()
tgt_tokenizer = BasicTokenizer()

src_sentences = [
    item["zh"]
    for item in train_data
]

tgt_sentences = [
    item["en"]
    for item in train_data
]

src_tokenizer.build_word_sheet(
    src_sentences,
    min_frequency=2
)

tgt_tokenizer.build_word_sheet(
    tgt_sentences,
    min_frequency=2
)

print(
    "src vocab:",
    src_tokenizer.vocab_size
)

print(
    "tgt vocab:",
    tgt_tokenizer.vocab_size
)



# ==================================================
# 6. 创建 Dataset
# ==================================================

train_dataset = TranslationDataset(
    data_list=train_data,
    src_tokenizer=src_tokenizer,
    tgt_tokenizer=tgt_tokenizer,
    max_src_len=64,
    max_tgt_len=64
)



# ==================================================
# 7. 创建 DataLoader
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


# ==================================================
#  创建 Transformer
# ==================================================

model = Transformer(
    src_vocab_size=src_tokenizer.vocab_size,
    target_vocab_size=tgt_tokenizer.vocab_size,
    d_model=512,
    num_head=8,
    high_dimension=1024,
    num_layers=3,
    pad_id=src_tokenizer.PAD_ID,
    dropout=0.1
)

model = model.to(device)

criterion = nn.CrossEntropyLoss(
    ignore_index=tgt_tokenizer.PAD_ID
)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

path = 'transformer_checkpoint.pt'
if os.path.exists(path):
    start_epoch, _ = load_checkpoint(model, optimizer, path, device=device)
else:
    start_epoch = -1

epochs = 20
print("=============loss start=============")

for epoch in range(start_epoch + 1, epochs):
    loss_avg = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"loss{epoch}", loss_avg)
    save_checkpoint(model,optimizer,epoch,loss_avg, "transformer_checkpoint.pt")

print("=============loss end=============")

print("==========================")
print("训练完成")
print("==========================")

print("\n==========================")
print("🧠 训练完成！开启人工审核模式")
print("==========================")


# 找两句你训练集里的话来测一测它有没有“记住”
test_sentences = [
    "中国 经济 经过 二十 年 快速 稳定 发展",
    "目前 经济 出现 了 通货 紧缩 的 趋向 " # 替换成你 JSONL 里的真实中文短句
]

for zh_sentence in test_sentences:
    en_translation = beam_search_translate_sentence(
        model=model,
        sentence=zh_sentence,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_len=64,
        device=device
    )
    print(f"🇨🇳 中文: {zh_sentence}")
    print(f"🇬🇧 机器翻译: {en_translation}")
    print("-" * 30)

