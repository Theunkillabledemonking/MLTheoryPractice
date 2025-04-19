import numpy as np

# ==== 1. 데이터 생성 ====
num_samples = 10
num_feauters = 2

np.random.seed(42) # 결과 재현성을 위한 시드 고정

# 입력 데이터 (10, 2) 크기의 배열
X = np.random.rand(num_samples, num_feauters) * 10

# 실제 가중치와 편향
true_weights = [3, 5]
true_bais = 10

# 노이즈 생성
noise = np.random.randn(num_samples) * 0.5

# 정답값 y 계산: y = 3*1 + 5*2 + 10 + noise
y = X[:, 0] * true_weights[0] + X[:, 1] * true_weights[1] + true_bais + noise

# ==== 2. 초기 파라미터 설정 ====
weights = np.random.rand(num_feauters) # 학습할 가중치
bais = np.random.rand()                # 학습할 편향

# ==== 3. 하이퍼파라미터 설정 ====
learning_late = 0.01
epochs = 10000
print_interval = 500 # 주기적으로 손실 출력

# ==== 4. 학습 시작 ====
for epoch in range(epochs):
    total_loss = 0.0
    total_weight_grand = np.zeros(num_feauters)
    total_bais_error = 0.0

    # 각 데이터 샘플에 대해 예측 및 기울기 계산
    for x_sample, y_true in zip(X, y):
        prediction = 0.0

        # 예측값 계산: w1*x1 + w2*x2 + ... + b
        for i in range(num_feauters):
            prediction += weights[i] * x_sample[i]
        prediction += bais

        # 오차 계산
        error = prediction - y_true

        # 가중치에 대한 gradient snwjr
        for i in range(num_feauters):
            total_weight_grand[i] += x_sample[i] * error

        # 편향 오차 누적
        total_bais_error += error

        # 손실 누적
        total_loss += error ** 2

        # 평균 gradient로 파라미터 업데이트
        for i in range(num_feauters):
            weights[i] -= learning_late * (total_weight_grand[i] / num_samples)
        bais -= learning_late * (total_bais_error / num_samples)

        # 일정 주기로 평균 손실 출력
        avergae_loss = total_loss / num_samples
        if epoch % print_interval == 0:
            print(f"[Epoch {epoch}] Average: Loos: {avergae_loss:.4f}")

# ==== 5. 학습 결과 출력 ====
print("\n Training Complete")
print("Learned Weights:", weights)
print("Learned Bias:", bais)

# ==== 6. 예측 결과 비교 ====
print ("\nPrediction vs Actual (first 5 samples)")
for i in range(5):
    prediction = 0.0
    for j in range(num_feauters):
        prediction += weights[j] * X[i][j]
    prediction += bais
    print(f"Predicted: {prediction:.2f} | Actual: {y[i]:.2f}")