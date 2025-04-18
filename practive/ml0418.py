import numpy as np


np.random.seed(5) # 시드값 고정

x = np.array([[1, 2], [3, 4]])
y = np.array([[2],[3]])

print(x.shape)
print(y.shape)

print(f"{x}\n{y}\n{x@y}")