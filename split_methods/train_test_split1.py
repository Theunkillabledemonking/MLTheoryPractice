from sklearn.model_selection import train_test_split
import numpy as np

# 1. 데이터셋 생성
# x: 10개의 샘플, 각 샘플은 2개의 특성(feature)을 가짐
# y: 각 샘플에 대한 이진 클래스(0 또는 1) 라벨
X = np.random.rand(10, 2)
y = np.random.randint(0, 2, size=10) # x와 샘플 수 맞춰야 오류 없음

# 2. 데이터셋 분할
# 전체 샘플 10개 중 20%인 2개는 테스트 세트, 나머지 8개는 훈련세트로 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 
)

# 3. 분할 결과 출력
print(f"[x] 전체 샘플 수: {X.shape[0]}, 특성 수 {X.shape[1]}개")
print(f"[y] 전체 라벨 수: {y.shape[0]}개\n")

print(f" 훈련 입력 데이터 (X_train): {X_train.shape[0]}개 샘플")
print(f" 테스트 입력 데이터 (X_test): {X_test.shape[0]}개 샘플")
print(f" 훈련 라벨 데이터 (y_train): {y_train.shape[0]}개 라벨")
print(f" 테스트 라벨 데이터 (y_test): {y_test.shape[0]}개 라벨")

# 4. 데이터 형태의 자료형도 출력
print(f"데이터 shape의 자료형: {type(X_train.shape)}")