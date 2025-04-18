import numpy as np

num_feature = 4
num_samples = 1000

np.random.seed(5) # 시드값 고정

X = np.random.rand(num_samples, num_feature) * 2 # 1000 * 4 -> 0~1의 미만 수
w_true = np.random.randint(1, 11, (num_feature, 1))
b_true = np.random.randn() * 0.5 # 1.5 ~ - 0.5 사이의 노이즈 값

y = X @ w_true + b_true

# w = np.ones(num_feature, 1) # 초기의 w값은 같으면 안됨
w = np.random.rand(num_feature, 1)
b = np.random.rand()

gradient = np.zeros(num_feature)
learning_late = 0.01

####################################################

for _ in range(1000):

    # 예측값 
    predict_y = X @ w + b

    # 오차
    error = predict_y - y

    # 기울기
    gradient_w = X.T @ error / num_samples
    gradient_b = error.mean()

    # 파라미터 업데이트
    w = w - learning_late * gradient_w
    b = b - learning_late * gradient_b

print(f"W: {w}, B: {b}")