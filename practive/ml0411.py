
samples = []
y = []
learning_late = 0.01

w = [0.2, 0.3]
b = 0.1

gradient_w = [0.0, 0.0]
gradient_b = 0.0

# Number of Sample : 1 epoch 
for dp, y_ in zip(samples, y):
    # 예측값
    predict_y = w[0] * dp[0] + w[1] * dp[1] + b

    # Erorr : 예측값 - 정답값
    error = predict_y - y_
    
    # Weight 기울기 값 누적
    gradient_w[0] += dp[0] * error
    gradient_w[1] += dp[1] * error

    # bias 기울기 값 누적
    gradient_b += error

# gradient updated of each W
w[0] = w[0] - gradient_w[0] / len(samples)
w[1] = w[1] - gradient_w[1] / len(samples)

# gradient updated of b
b = b - gradient_b / len(samples)
