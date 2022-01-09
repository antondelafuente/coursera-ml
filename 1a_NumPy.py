import numpy as np

def dJ(X, y, theta):
    return (X @ theta - y) @ X / len(y)

def gradientDescent(X, y, theta, alpha, num_iters):
    for i in range(num_iters):
        theta -= alpha * dJ(X, y, theta)
    return theta

data = np.loadtxt('ex1data1.txt', delimiter=',')
x, y = data[:, 0], data[:, 1]
X = np.column_stack((np.ones(len(x)), x))

theta = gradientDescent(X, y, np.zeros(X.shape[1]), 0.01, 10000)
print(theta)

theta = np.linalg.inv(X.T @ X) @ X.T @ y
print(theta)
