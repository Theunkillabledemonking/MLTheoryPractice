from sklearn.model_selection import train_test_split
import numpy as np

# 1. 더미 데이터 생성
# - X: 총 100개의 샘플,
#   각 샘플은 3개의 특성(feature)을 가짐
# - y: 각 샘플의 이진 분류 라벨 (0 또는 1)
X = np.random.rand(100, 3)
y = np.random.randint(0, 2, size=100)

# 2. 데이터셋 1차 분할
# - 전체 데이터의 70%를 훈련용,
#   30%를 임시(validation/test)용으로 분할
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. 데이터셋 2차 분할
# - 임시 데이터셋(30%)을 다시 50:50으로 나누어
#   검증용과 테스트용으로 분할
# - 결과적으로 전체 기준으로는 각 15%씩 차지
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# 4. 분할된 데이터셋 크기 출력
# - 각 데이터셋이 잘 나뉘었는지 shape을 통해 확인
print("훈련 데이터 크기: ", X_train.shape)  # (70, 3) -> 전체의 70%
print("검증 데이터의 크기:", X_val.shape)   # (15, 3) -> 전체의 15% 
print("테스트 데이터의 크기:", X_test.shape)   # (15, 3) -> 전체의 15% 
