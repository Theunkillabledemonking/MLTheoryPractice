import numpy as np
import matplotlib.pyplot as plt

# Data set
# input
# input data, features
# H(x) -> input data : x1 -> xn
x_train = [ np.random.rand() * 10 for _ in range(50) ]
y_train = [ val * np.random.rand() * 5 for val in x_train]

# BGC (Batch Gradient Descent) 배치하강법을 이용하여 Linear Regression 적용


plt.scatter(x_train,y_train, color = "blue")
plt.show()
# output
# label
# H(x1) -> H(x2) 