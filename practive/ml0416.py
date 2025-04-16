import numpy as np

# h(w) = wx1 + wx2 + wx3

num_features = 3
num_samples = 30

np.random.seed(1)
np.set_printoptions(suppress=True, precision=3)

X = np.random.rand(num_samples, num_features)

w_true = np.random.randint(1, 10, num_features)
b_true = np.random.randn() * 0.5

y = X @ w_true + b_true

# Learning
w = np.random.rand(num_features)
b = np.random.randn()
learning_late = 0.01
epchos = 100000

gradient = np.zeros(num_features)

for epcho in range(epchos):


    # prediction
    prediction = (X @ w) + b

    # error
    error = (prediction - y)

    # gradient
    gradient = (X.T @ error) / num_samples

    # update
    w = w - learning_late * gradient
    b = b - learning_late * error.mean()
    print(w)
    print(b)

print(f"w_ture:, {w_true}, b_true: {b_true}")
print(f"w, {w}, b: {b}")