#!/Users/gwan/Desktop/Code/myenv/bin/python3
import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(50, 1) * 10 # 0.0 ~ 10.0
y = [2 * val + np.random.rand() * 4 for val in x]

# x1 -> x4, y1 -> y4
loss = 0.0
w = 4
learning_rate = 0.01
avg = []

# w ------ > ?
for _ in range(20):
    slope_sum = 0
    loss = 0
    for x_train, y_train in zip(x, y):
        # X1 -> Y1

        # 현 W에 대한 X1의 예측값
        pred_val = w * x_train
        
        # 손실 값 (예측값 - 정답) + 
        loss += (pred_val - y_train)**2

        slope_sum += x_train * (w * x_train - y_train)

    # w  업데이트 -> 경사 하강법
    w = w - learning_rate * (slope_sum - len(x))

    # 평균 -> cost function
    avg.append(loss/len(x))

print(f"W: {w}")

# H (x) = 2*x
# w -> 2
plt.scatter(x, y)
plt.grid()
plt.plot([val for val in x], [w*y_val for y_val in x])
plt.show()