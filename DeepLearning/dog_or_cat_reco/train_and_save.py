from fastai.vision.all import *

# 1. 强制 M1 GPU 加速 (推理其实 CPU 也行，但咱们有 GPU 就用上)
default_device(torch.device('mps'))

# 2. 加载模型 (请确保文件已经搬到了你的项目目录下)
model_file = 'my_cat_dog_model.pkl'
learn_inf = load_learner(model_file)

# 3. 找一张图！(你可以去网上下一张，或者直接用刚才数据文件夹里的一张)
# 比如：img_path = '/Users/ms/.fastai/data/oxford-iiit-pet/images/Abyssinian_1.jpg'
img_path = 'img.png'  # 换成你下载的图片名

# 4. 执行预测
label, _, probs = learn_inf.predict(img_path)

print(f"\n=======================")
print(f"AI 鉴定结果: 这{'是' if label == 'True' else '不是'}一只猫")
print(f"确信度(猫的概率): {probs[1].item():.4%}")
print(f"=======================")