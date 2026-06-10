# 你现在是某头部电商平台的 AI 算法工程师。运营部门急疯了：最近有一批老用户突然不活跃了。
# 他们把上个月的用户行为数据导给了你，希望你训练一个“用户流失雷达”。
# 只要雷达探测到某个用户流失概率超过 50%，运营就会立刻给他发优惠券强行挽留！

# 这份数据集包含了 1000 行样本，具体特征含义如下：
#
# user_id: 用户标识（提示：在训练前记得用 df.drop('user_id', axis=1) 把它剔除，它不是特征！）

# age: 年龄
# days_since_last_login: 距离上次登录的天数
# avg_monthly_spend: 月均消费额
# total_spend_ytd: 今年累计消费额
# customer_service_calls: 客服投诉次数
# has_premium: 是否为高级会员 (1=是，0=否)

# churn: 目标变量，是否流失 (1=流失，0=留存)



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import joblib

np.set_printoptions(suppress=True, precision=4)

# 1. 读取数据
df = pd.read_csv('ecommerce_churn_large.csv')



# 2. 剥离无用列和分离目标变量 X, y
y = df['churn']
X = df.drop(['user_id', 'churn'], axis=1)

# 3. 三库隔离：按 80% 训练，20% 测试进行切分
# TODO: 使用 train_test_split (设置 random_state=42 保证结果可复现)
# 训练集和测试集的拆分， train_test_split 支持 DataFrame 或者 numpy 的矩阵， 但是 DataFrame 的结果会携带 下标 和列头部
# 在排查问题是更方便

# test_size=0.2 表示 20% 的数据用来测试，80% 用来训练
# random_state=42 是一个“随机种子”。填上它，保证你每次运行切出来的数据都是同一批，方便排查 Bug。
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)



# 4. 搭建防呆流水线 Pipeline
# 尝试使用 L2 正则化，你可以先用默认参数，也可以试着调节 C 值 (如 C=0.01 或 C=10)
# TODO: 编写 make_pipeline
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty='l2', C = 0.001)
)

# 5. 训练并在测试集上打分
# TODO: 调用 fit 和 predict，并使用 accuracy_score 查看准确率

pipeline.fit(X_train, y_train)

# 可以查看下训练的结果
model = pipeline.named_steps['logisticregression']
b = model.intercept_[0]
a = model.coef_[0]
features_names = X_train.columns
weight_df = pd.DataFrame({
    '特征名称': features_names,
    '权重值 (w)': a
})
weight_df = weight_df.sort_values(by = '权重值 (w)', ascending = False)


# 预测
predictions = pipeline.predict(X_test)

# 查看准确率
acc = accuracy_score(y_test, predictions)
print(acc)
report = classification_report(y_test, predictions)
print("详细分类报告：\n", report)


joblib.dump(pipeline, 'model_v1.pkl')