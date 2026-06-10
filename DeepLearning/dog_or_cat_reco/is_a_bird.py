from fastai.vision.all import *
import os
from pathlib import Path

# 1. 强制挂载 M1 GPU 引擎
default_device(torch.device('mps'))
print("🚀 引擎点火成功，正在连接高带宽服务器下载数据...")

# 2. 绕过爬虫！直接从官方 CDN 下载极其干净的『牛津宠物数据集』(约 800MB)
path = untar_data(URLs.PETS)/'images'
# ~/.fastai/data/oxford-iiit-pet/images


# 3. 极简打标逻辑：首字母大写的是猫，小写的是狗
def is_cat(x): return x.name[0].isupper()


# 🧱 核心来了：用 DataBlock 手工拼装流水线！
pets_block = DataBlock(
    # 1. 定义输入和输出类型：输入是图片(ImageBlock)，输出是分类(CategoryBlock)
    blocks=(ImageBlock, CategoryBlock),

    # 2. 告诉它怎么拿到所有数据：去路径下把所有图片文件抓出来
    get_items=get_image_files,

    # 3. 告诉它怎么切分数据：随机抽出 20% 作为验证集，留 80% 训练
    splitter=RandomSplitter(valid_pct=0.2, seed=42),

    # 4. 告诉它怎么给数据打标签：用我们上面写的 is_cat 函数
    get_y=is_cat,

    # 5. 告诉它进入 GPU 之前的变形操作：把所有图片压缩成 224x224
    item_tfms=Resize(224)
)


# 4. 构建数据装载机 (⚠️ M1 专属 8GB 内存防爆补丁：bs=16)
dls = pets_block.dataloaders(path, bs=16)

# 查看装载的 随机6张图片
# dls.show_batch()
# plt.show()


# 5. 下载预训练模型，准备大干一场
print("🧠 正在加载 ResNet18 神经网络...")
learn = vision_learner(dls, resnet18, metrics=error_rate)

# 6. 开始微调！(你的电脑可能会微微发热)
print("🔥 训练开始！请盯紧下方的 error_rate...")
learn.fine_tune(1)

# 7. 训练结束，恭喜通关！
print("🎉 恭喜！你的 M1 刚刚完成了一次真实的深度学习训练！")



# 1. 获取当前这个 predict.py 文件所在的文件夹绝对路径
current_dir = Path(__file__).parent.absolute()

# 2. 拼接文件名
# 这样无论你在哪里点击运行，它永远会去脚本同级目录下找文件
model_path = current_dir / 'my_cat_dog_model.pkl'

learn.export(model_path)

# 打印出它到底存到硬盘哪个角落了
print(f"✅ 模型已存盘！")
