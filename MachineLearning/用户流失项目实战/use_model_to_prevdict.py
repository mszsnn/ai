import joblib
import pandas as pd

loaded_model = joblib.load('model_v1.pkl')

frontend_payload = {
    'age': 35,
    'days_since_last_login': 40,
    'avg_monthly_spend': 100.0,
    'total_spend_ytd': 250.0,
    'customer_service_calls': 3,
    'has_premium': 1
}

# 假设这是从数据库里拿到的一个新用户的最新数据
new_user_data = pd.DataFrame([frontend_payload])


# 1. 自动执行 StandardScaler(内部已存好均值标准差) -> 逻辑回归预测
# 直接拿到 0 或 1
originis_churn = loaded_model.predict(new_user_data)

# 2. 或者拿到具体的流失概率（通常运营更需要这个）
origin = loaded_model.predict_proba(new_user_data)

print(origin)


# 行全都要， 只要第二列
churn_prob = origin[:, 1]

print(f"该用户流失概率为: {churn_prob[0]:.2%}")