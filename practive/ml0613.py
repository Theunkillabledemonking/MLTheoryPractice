from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터셋 로딩 및 분할
digits = load_digits() # 사이킷런에 저장된 data set
features = digits.data                    # (1797, 64): 8x8 이미지 벡터
labels = digits.target                    # (1797,): 0~9 클래스 정수

print(features.shape)
print(labels.shape)
print(labels[:100])

# 2. 학습/테스트 셋 분할
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels # 비율를 0~9 동일한 비율로 유지
)

# 3. 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 4. 기본 설정
num_featuers = X_train_std.shape[1] # 특성의 개수 64개
num_samples = X_train_std.shape[0]  # 샘플의 개수 1437개
num_class = 10

W = np.random.randn(num_featuers, num_class) # 정규분포 (확률분포 난수값 발생)
b = np.zeros(num_class) # 바이어스


print(W.shape)