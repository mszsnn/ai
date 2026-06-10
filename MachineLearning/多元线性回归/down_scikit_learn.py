import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd
# ==========================================
# 1. 准备数据
# ==========================================
# 原始数据：[面积, 卧室, 楼层, 房龄]


raw_data = {
    'Area(sqft)': [2104, 1416, 852, 1930, 1200, 1534, 1000, 2400],  # 面积
    'Bedrooms':   [5, 3, 2, 4, 3, 3, 2, 4],                # 卧室数
    'Floors':     [1, 2, 1, 2, 1, 2, 1, 2],   # 楼层数
    'Age(years)': [45, 40, 35, 15, 20, 30, 10, 5],     # 房龄
    'Price(1000$)':[460, 232, 178, 315, 200, 270, 190, 500]  # 真实的售价 (标签)
}


df = pd.DataFrame(raw_data)

X = df.drop('Price(1000$)', axis=1).to_numpy()
y = df['Price(1000$)'].to_numpy()

# ==========================================
# 2. 构建自动化流水线 (Pipeline)
# ==========================================
# 在工业界，我们通常用 Pipeline 将预处理和模型绑死在一起，防止泄露或遗漏
model_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False), # 第一步：自动生成二次多项式特征
    StandardScaler(),                                 # 第二步：自动执行 Z-Score 标准化
    LinearRegression()                                # 第三步：送入线性回归引擎
)

# ==========================================
# 3. 一键训练模型 (Fit)
# ==========================================
print("开始训练模型...")
# 这一行代码内部自动完成了：算多项式 -> 算 mu 和 sigma 并缩放 -> 求解最优权重 W 和 b
model_pipeline.fit(X, y)
print("训练完成！\n")


# ==========================================
# 4. 预测新数据 (Predict)
# ==========================================
# 新房源数据：1200平方英尺, 3个卧室, 1层, 40年房龄
x_new_house = np.array([[1200, 3, 1, 40]])

# 极其爽快：不需要你手动调用 transform 缩放，流水线会自动处理！
predicted_price = model_pipeline.predict(x_new_house)

print(f"🏠 新房预测价格: ${predicted_price[0]:.2f}k")

# ==========================================
# 5. 偷窥模型内部的 "脑子" (可选)
# ==========================================
# 我们把流水线里的 LinearRegression 引擎单独提出来
linear_engine = model_pipeline.named_steps['linearregression']

print("\n--- 模型内部参数 ---")
print(f"偏置 b (截距): {linear_engine.intercept_:.2f}")
# 因为生成了多项式，这里的权重 w 数量会比原来多很多
print(f"权重 w (系数): {linear_engine.coef_}")