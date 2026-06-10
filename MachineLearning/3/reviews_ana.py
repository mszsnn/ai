from unicodedata import category

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
import sys

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# 这些代码是为了理解分词在具体做什么， 核心代码在最下面
# 创建 计数向量器
vectorizer = CountVectorizer()

orign_data = pd.read_csv('advanced_product_reviews_with_stars.csv')

# 获取分词矩阵
x = vectorizer.fit_transform(orign_data['review'])

# 获取 分词列表
feature_names = vectorizer.get_feature_names_out()

# 假设你有一条评论：
#
# "great product great"
#
# 经过 CountVectorizer 后：
#
# 👉 词表：
#
# feature_names = ["great", "product"]
# 稀疏表示（核心）
# row.indices = [0, 1]   # 第0列，第1列
# row.data    = [2, 1]   # 出现次数
# ✅ 三、zip 在干嘛？
# list(zip(row.indices, row.data))
#
# 👉 结果：
#
# [(0, 2), (1, 1)]

# 构建 list  然后扔到 列表中
word_counts = []

for row in x:
    rowData = {
        feature_names[i]: count   # 注意 这里没有 , 因为这是一个表达式 + 循环规则
        for i, count in zip(row.indices, row.data)
    }
    word_counts.append(rowData)

orign_data['count'] = word_counts


# 用柱状图来表示 分类
# category_counts = orign_data['product_name'].value_counts()
# print(category_counts.index[0])

# 这里 plot 底层内部其实调用了 plt
# 这是绘制分类的柱状图， 得出了  Stainless Steel Pan
# plt.bar(category_counts.index, category_counts.values )
# category_counts.plot(kind='bar')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()



# 这是用 用字典构造的模型和上面用 文本分词构建的模型， 本质上是一样的, 因为 本身字典也是从 文本分词中构造出来的， 所以结果完全一样
# vec = DictVectorizer()
# vec.fit(orign_data['count'])
# vec_features_names = vectorizer.get_feature_names_out()
# print(vec_features_names)


# 接下来构建模型


# 首先进行数据分类，分开训练集和测试集，  sample 和 train_test_split 的用法以及特征对比

# 如果用 sample,那就需要自己手动分，并且需要对其 index

# train = orign_data.sample(frac = 0.8, random_state=0)
# test = orign_data.drop(train.index)
# x_train = train['review']
# y_train = train['sentiment']
# x_test = test['review']
# y_test = test['sentiment']

# 如果用 train_test_split, 基本上一句话搞定， 最最关键的是
# 1 支持其他特征
# 2 支持分层抽样， 也就是保证 训练集和测试集的正负比例是一致的， 先进行按照结果分类， 然后从每个类型中进行 比例划分



orign_data['sentiment'] = orign_data['stars'].apply(lambda x: 1 if x > 3 else 0)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    orign_data['sentiment'],
    test_size=0.2,
    random_state=0,
)

# 迭代次数
model = LogisticRegression(max_iter=500)

# 进行训练
model.fit(x_train, y_train)

# 用模型的结果预测测试结合
y_pred = model.predict(x_test)

# 计算误差
print('accuracy', accuracy_score(y_test, y_pred))


# 1 更加具体的迭代过程， 因为之前的 线性回归是明显具有最优解的， 可以通过数学计算出来，而这里具体是如何迭代的




