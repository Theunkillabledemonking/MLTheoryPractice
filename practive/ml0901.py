w1 = 0.5
w2 = 0.5
bias = -0.5 # - AND , + OR operator

def perceptron(x1, x2):
    netInputValue = x1*w1 + x2*w2 + bias
    return 1 if netInputValue > 0 else 0

print(perceptron(0, 0))
print(perceptron(0, 1))
print(perceptron(1, 0))
print(perceptron(1, 1))
