import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor # SG Linear regerssion
from sklearn.metrics import mean_squared_error # 평균 제곱 오차

np.set_printoptions(suppress=True, precision=2)
X = np.random.rand(100, 1) * 10
# H(x) = w * x + b
y = 2.5 * X + np.random.randn(100, 1) * 2  # (  -3 ~ 3) * 2 = -6 ~ 6
y = y.ravel() # SGDRegressor는 1차원 타겟값을 요구

# 모델 생성 후 하이퍼파라미터 설정
model = SGDRegressor(
    max_iter=1000, # 학습 반복 횟수 (epoch 수)
    learning_rate="constant",
    eta0=0.001,
    penalty=None,
    random_state=0
)

# 학습 실시
model.fit(X, y)

# 평가
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)

print(y_pred)
print(y)
print(mse)