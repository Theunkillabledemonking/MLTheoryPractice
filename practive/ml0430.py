
from sklearn.datasets import fetch_california_housing

# 1. 데이터 로드
data = fetch_california_housing()

#2. 주요 속성 확인
X = data.data # 입력 데이터 (numpy.ndarray)
y = data.target # 타겟값 (중간 집값, 단위: 100,000$)
feature_names = data.feature_names # 특성 이름 리스트

print("입력 X shape:", X.shape)  # (20640, 8)
print("타겟 y shape:", y.shape)  # (20640,)
print("특성 이름:", feature_names) # ['MedInc', 'House]
print("설명:", data.DESCR[:1000]) # 데이터셋 설명 일부 출력