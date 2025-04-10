import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor  # SGD Linear Regression
from sklearn.metrics import mean_squared_error # 평균 제곱 오차 MSE

# x: 0 ~ 10 사이 무작위 수 100개
# y: 2.5x + 약간의 노이즈
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + np.random.randn(100, 1) * 2
y = y.ravel() # SGDRegressor는 1차원 타겟값을 요규

model = SGDRegressor(max_iter=1000, # 학습 반복 횟수 (epoch 수)
                     learning_rate='constant',
                     eta0=0.01,     # 고정 학습률
                     penalty=None,  # 정규화 없음
                     random_state=0
)

model.fit(X, y) # 모델 학습
y_pred = model.predict(X) # 예측 값 출력 (일반화)

mse = mean_squared_error(y, y_pred)
print(f"평균 제곱 오차(MSE): {mse:.4f}")

# 회귀선 그리기
x_line = np.linspace(0, 10, 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.scatter(X, y, label='Data', alpha=0.6)
plt.plot(x_line, y_line, color='red', label='SGD Regression Line')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression With SGDRegressor")
plt.legend()
plt.grid(True)
plt.show()