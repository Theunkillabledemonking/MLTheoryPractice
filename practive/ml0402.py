from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# X : 입력 값
# Y : 출력 값 (정답)
x = np.random.rand(10, 2) * 5 # 5 * 2 매트릭스 생성, 랜덤값 0에서 5미만 값 설정
y = np.random.randint(0, 2, size=10)

# for zip(x, y):


# sample -> 100
# train -> 80
# test -> 20

X_train, X_test, Y_train, Y_test = \
    train_test_split(x, y, test_size=0.2, random_state=1)
print(f"X_train.shape: {X_train.shape}")
print(X_train)

print(f"X_test.shape: {X_test.shape}")
print(X_test)

print(f"Y_train.shape: {Y_train.shape}")
print(Y_train)

print(f"Y_test.shape: {Y_test.shape}")
print(Y_test)