import numpy as np

# 특성(feature) 수
num_feature = 4
# 샘플(sample) 수
num_samples = 1000

# 시드 고정 (난수 고정)
np.random.seed(42)

# 입력 데이터 X 생성 (1000,4) 행렬, 0~2 범위의 값
X = np.random.rand(num_samples, num_feature) * 2

# 진짜 가중치 w_true 생성: (4, 1)
w_true = np.random.randint(1, 11, (num_feature, 1))

# 진짜 편향 b_true 생성: 스칼라 (float 하나)
b_true = np.random.randn() * 0.5

# 정답 데이터 y 생성: (1000, 1) = (1000,4) @ (4,1) + (스칼라)
# [벡터와 스칼라 연산 1]: (X @ wtrue) 결과 (1000, 1) + b_true(스칼라) 
y = X @ w_true + b_true

# ------------------------ 학습용 변수 초기화 -----------------------------

# w를 무작위 값으로 초기화 (4, 1) 행렬
w = np.random.rand(num_feature, 1)

# b를 무작위 값으로 초기화: 스칼라 (float 하나)
b = np.random.rand()

# gradient 초기화: (4,) 크기 벡터 (나중에 사용하지 않음)
gradient = np.zeros(num_feature)

# 학습률
learning_late = 0.01

####################################################

for _ in range(1000):

    # 예측값 predict_y 계산
    # [백터와 스칼라 연산2]: (X @ w) 결과 (1000, 1) + b(스칼라)
    predict_y = X @ w + b

    # 오차(error) 계산: 예측 - 정답
    # [벡터 연산]: (1000, 1) - (1000, 1) = (1000, 1)
    error = predict_y - y

    # 가중치에 대한 기울기 gradient_w 계산
    # [벡터 연산]: (X.T @ error) -> (4, 1000) @ (1000, 1) = (4,1)
    gradient_w = X.T @ error / num_samples
    
    # 편향에 대한 기울기 gradient_b 계산
    # [벡터 연산]: error.mean() -> (1000, 1)의 평균값 하나
    gradient_b = error.mean()


    # 파라미터 업데이트 (경사하강)
    # [벡터 연산]: (4,1) - (4,1)
    w = w - learning_late * gradient_w

    # [스칼라 연산]: (스칼라) - (스칼라)
    b = b - learning_late * gradient_b

# 최종 학습된 파라미터 출력 
print(f"W: {w}, B: {b}")