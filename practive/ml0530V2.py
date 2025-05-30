from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 110)

max = x.max()
min = x.min()

print(max, min)

values = [ (item - min) / (max - min) for item in x ]
print(values)

one_hot = np.eye(4)
print(one_hot, "\n\n")

y_list = [0, 1, 0, 3, 2, 3]
one_hot_value = one_hot[y_list] # One-hot Encoding을 함.

print(one_hot_value)

print(np.argmax(one_hot_value, axis=1)) # One-hot Encoding한 값을 디코딩하여 복원시키는ㄱ ㅓㅅ

x = np.array([[2, 2, 1], \
              [2, 1, 1], \
              [2, 1, 1]])

print(x.sum(0)) # 0은 열 1은 행