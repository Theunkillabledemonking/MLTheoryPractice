from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# 유방암 환자의 훈련 데이터 셋 Load -> 자료형 Bunch Class
cancer = load_breast_cancer()

# 훈련데이터는 Input 데이터와 Target 데이터로 구성
# Input 데이터 shape: center.data.shape
# Target 데이터 Shape: center.target.shape
print(cancer.data.shape, cancer.target.shape)

# Input Data Feature
print(cancer.feature_names)

# input Data 1~3행 출력
print(cancer.data[0:5])

# Target Data 1~100 행 출력
print(cancer.target[:100])

plt.scatter(cancer.data[:, 0], cancer.target)
plt.show()