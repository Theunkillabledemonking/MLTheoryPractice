import numpy as np

# ==== 1. 데이터 생성 ====
np.random.seed(42)
num_samples = 10
num_features = 2

X = np.random.rand(num_samples, num_features) * 10 # 특성 2, # 샘플 10
true_w = np.array([3, 5]) # 2행 1열
true_b = 10
noise = np.random.randn(num_samples) * 0.5
y = X @ true_w + true_b + noise # 2행 10열 @  2행 1열 -> 10행 1열

# ==== 2. 파라미터 초기화
w = np.random.rand(num_features)
b = np.random.rand()
lr = 0.03
epochs = 10000


# ==== 3. 학습 루프 ====
for epoch in range(epochs):
    pred = X @ w + b        # 예측값 (벡터)
    error = pred - y        # 오차 (벡터)

    grad_w = X.T @ error / num_samples  # 가중치 gradient
    grad_b = error.mean()               # 편향 gradient
    
    w -= lr * grad_w
    b -= lr * grad_b

    if epoch % 50 == 0:
        loss = (error ** 2).mean()
        print(f"[Epoch {epoch}] 평균 손실: {loss:.4f}")

# ==== 4. 결과 ====
print("\n학습된 가중치:", w)
print("학습된 편향:", b)
