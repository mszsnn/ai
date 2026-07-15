import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from llm.cn2en.tokenizer import BasicTokenizer

from llm.cn2en.translation_dataset import (
    load_jsonl,
    split_raw_data,
    TranslationDataset
)

from llm.cn2en.transformer import Transformer



# ==================================================
# 1. 设备
# ==================================================

import torch

device = torch.device("cpu")

print(
    "device:",
    device
)



# ==================================================
# 2. 读取数据
# ==================================================

data = load_jsonl(
    "translation_dataset.jsonl"
)
print(
    "全部数据:",
    len(data)
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


print(
    "train:",
    len(train_data)
)

print(
    "val:",
    len(val_data)
)

print(
    "test:",
    len(test_data)
)



# ==================================================
# 4. 创建 Tokenizer
# ==================================================

src_tokenizer = BasicTokenizer()
tgt_tokenizer = BasicTokenizer()



# ==================================================
# 5. 只使用训练集构建词表
# ==================================================

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
# 8. 查看一个 batch
# ==================================================

# iter 将 train_loader 生层迭代器
# next() 取出下一个迭代器
src, decoder_input, labels = next(
    iter(train_loader)
)



print("==========================")
print("Batch检查")
print("==========================")


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



# 放入 GPU

src = src.to(device)
decoder_input = decoder_input.to(device)
labels = labels.to(device)

# ==================================================
# 9. 创建 Transformer
# ==================================================

model = Transformer(
    src_vocab_size=src_tokenizer.vocab_size,
    target_vocab_size=tgt_tokenizer.vocab_size,
    d_model=64,
    num_head=8,
    high_dimension=256,
    num_layers=2,
    pad_id=src_tokenizer.PAD_ID,
    dropout=0.1
)

model = model.to(device)
print("==========================")
print("模型创建成功")
print("==========================")

# ==================================================
# 10. Forward
# ==================================================

logits = model(
    src,
    decoder_input
)

print("==========================")
print("Forward")
print("==========================")

# [batch, seq_len, vocab_size]
print(
    "logits:",
    logits.shape
)
# ==================================================
# 11. Loss
# ==================================================

# 创建交叉损失熵函数
criterion = nn.CrossEntropyLoss(
    ignore_index=tgt_tokenizer.PAD_ID
)
#    logits.reshape(
#         -1,
#         logits.size(-1)
#     ) 意思就是 将分类全部展开， batch * seq_len 个预测，每个预测 vocab_size 种结果

# labels.reshape(-1)  意思是 每个预测的答案是 batch * seq_len 个， 然后两者对比

loss = criterion(
    logits.reshape(
        -1,
        logits.size(-1)
    ),
    labels.reshape(-1)
)

print("==========================")
print("Loss")
print("==========================")

print(
    "loss:",
    loss.item()
)
# ==================================================
# 12. Backward
# ==================================================
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

optimizer.zero_grad()
loss.backward()

print("==========================")
print("Gradient检查")
print("==========================")

grad = (
    model
    .encoder
    .embedding
    .embedding
    .weight
    .grad
)
print(
    "gradient:",
    grad.shape
)
# ==================================================
# 13. 参数更新
# ==================================================
optimizer.step()
print("==========================")
print("完成")
print("==========================")

