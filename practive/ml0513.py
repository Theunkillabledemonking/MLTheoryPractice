import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드 및 분할
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
learning_late = 0.01
# 2. 훈련/테스트 셋 분리 (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42 # stratify X 와 y 간의 동일한 비율 유지
) 

# 3. 특성 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

num_featers = X_train.shape[1] # 30

w = np.random.randn(num_featers, 1)
b = np.random.randn()
np.set_printoptions(suppress=True, precision=5)

# z = wx + b
z = X_train @ w + b
y_train = y_train.reshape(-1, 1) 

#3지정안하고 핼령을 마꾸어 선언

# prediction = 1 / (1 + e) 시그모이드 함수
prediction = 1/ (1 + np.exp(-z))

# Error  = predcition - y_train
error = prediction - y_train

# gradient_w, gradient_b 
gradient_w = X_train.T @ error / len(X_train)
gradient_b = error.mean()

print(gradient_w, gradient_w.shape)
print(gradient_b, gradient_b.shape)

# update parameters: w, b
w = w - learning_late * gradient_w
b = b - learning_late * gradient_b

# cacluate loss 출력
loss = -np.mean(
    y_train * np.log(prediction + 1e-15) +
    (1 - y_train) * np.log(prediction + 1e-15) 
)

