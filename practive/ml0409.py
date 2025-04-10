import numpy as np

num_of_sample = 5
num_of_features = 2

# data set
# H(x) = 5X + 3x + 4

np.random.seed(1)
np.set_printoptions(False, suppress=True)
X = np.random.rand(num_of_sample, num_of_features) * 10
x_true = [5, 3]
b_true = 4
noise = np.random.rand(num_of_sample) * 2
y = X[:, 0] * 5 + X[:, 1] * 3 + b_true + noise

print(X)
print(y)
# print(X[:, 0] * 5)
# print(X[:, 1] * 3)
# print(X[:, 0] * 5 + X[:, 1] * 3 + b_true)