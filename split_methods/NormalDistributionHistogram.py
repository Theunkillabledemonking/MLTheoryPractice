from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

x = [val for val in range(20)]


np.set_printoptions(suppress=True, precision=10)
scaler1 = StandardScaler() # 표준화
scaler2 = StandardScaler() # 표준화

values1 = np.array((160, 170, 190, 180)).reshape(-1, 1)
values2 = np.array((4000000000, 70000000, 2000000000, 30000000)).reshape(-1, 1)

fit_values1 = scaler1.fit_transform(values1)
fit_values2 = scaler2.fit_transform(values2) # 평균,분산,표준편차를 구하고 표준화로 변환

print(fit_values1)
print(fit_values2)

# print(fit_values1.mean_, fit_values1.var_, fit_values1.scale_) # 평균, 분산, 표준편차
# print(fit_values2.mean_, fit_values2.var_, fit_values2.scale_) # 평균, 분산, 표준편차
