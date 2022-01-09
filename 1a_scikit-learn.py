import numpy as np
from sklearn.linear_model import LinearRegression

def lin_reg_sklearn(x, y):
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    return np.append(lin_reg.intercept_, lin_reg.coef_)


data = np.loadtxt('ex1data1.txt', delimiter=',')
x, y = data[:, 0].reshape(-1, 1), data[:, 1]

theta = lin_reg_sklearn(x, y)
print(theta)
