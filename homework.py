import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
uploaded = files.upload()
file = list(uploaded.keys())[0]
data = pd.read_csv(file)
print(data.head())
print(data.columns)
X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values
m = 0
b = 0
learning_rate = 0.01
iterations = 1000
n = len(X)
for i in range(iterations):
    Y_pred = m * X + b
    dm = (2/n) * np.sum(X * (Y_pred - Y))
    db = (2/n) * np.sum(Y_pred - Y)
    m = m - learning_rate * dm
    b = b - learning_rate * db
print("Slope:", m)
print("Intercept:", b)
print("Regression Equation:")
print("Y =", m, "* X +", b)
plt.scatter(X, Y, color="blue", label="Data")
plt.plot(X, m * X + b, color="red", label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()
