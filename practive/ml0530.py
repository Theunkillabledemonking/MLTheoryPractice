from sklearn.preprocessing import StandardScaler
import numpy as np

x = np.arange(10)

x_sum = sum(x)
x_avg = x_sum / len(x) # 평균

# 분산 값
variance = 0.0
for item in x:
    variance += (item - x_avg)**2

variance /= len(x)
std = np.sqrt(variance) # 표준 편차

print(x_avg, variance, std)

np_avg = x.mean()
np_variance = x.var()
np_std = x.std()        # 표준 편차
print(np_avg, np_variance, np_std)

val = np.random.randn()
print(val)