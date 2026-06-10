import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_train = np.array([
    [1.2, 35], [1.5, 42], [1.8, 38], [2.0, 50],
    [4.5, 65], [5.0, 70], [5.5, 58], [6.2, 80]
])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])


# 打造流水线

pipeline = Pipeline(
    StandardScaler(),
    LogisticRegression(penalty=None)
)

# 3. 一键训练 (黑魔法在这里！)
# X_train 是原始数据，完全不需要你手动缩放！
# pipeline 会自动先把它喂给 StandardScaler 缩放，然后再把缩放后的数据喂给 LogisticRegression 训练。
pipeline.fit(X_train, y_train)

# 4. 一键预测上线
X_new = np.array([[1.3, 30], [6.0, 75]])

# 直接传入原始的新数据！
# pipeline 会自动调取刚才存好的均值和标准差对其进行缩放，然后预测。
predictions = pipeline.predict(X_new)

print(f"预测类别: {predictions}")