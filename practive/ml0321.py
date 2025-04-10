import numpy as np
import matplotlib.pyplot as plt

# input
# input data, features
# H(x) -> input data : x1 -> xn
x_train = [ np.random.rand() * 10 for _ in range(50)]
y_train = [val + np.random.rand() * 5 for val in x_train]

# BGC (Batch Gradient Descent) 배치경사하강법을
# 이용하여 Linear Regresion 적용

w = 0.0
b = 0.0
learning_rate = 0.01
epoch = 50
loss_history = []
# H(x) -> w * x + b

for num_of_epoch in range(epoch):
    gradient_w_sum = 0.0
    gradient_b_sum = 0.0
    loss = 0.0
    # GD 수행 후 최적의 w 값 도출
    for x, y in zip(x_train, y_train):
        # 기울기 값을 누적 loss 값을 구할 필요는 X
        gradient_w_sum += x * (w * x + b - y) # coss function의 기울기  
        gradient_b_sum += (w * x + b - y)
        loss += (w * x + b - y) ** 2

    # w 값 업데이트
    w = w - learning_rate * (gradient_w_sum / len(x_train))
    b = b - learning_rate * (gradient_b_sum / len(x_train))

    loss_history.append(loss/len(x_train))
    print(f"Epoch {num_of_epoch + 1}, Loss: {loss / len(x_train)}")


x_test = [val for val in range(20)]
y_test = [w * val + b for val in x_test]

fig, axe = plt.subplots(1, 2)
axe[0].scatter(x_train,y_train, color="blue")
axe[0].plot(x_test, y_test)
axe[1].plot([val for val in range(epoch)], loss_history)
plt.show()

# output
