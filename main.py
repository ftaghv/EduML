# main.py
import numpy as np
from linear_model.linear_regression import LinearRegression

X = np.array([[1,2],[2,3],[3,4]])
y = np.array([3,5,7])

model = LinearRegression(epoch=100)
model.fit(X, y)

y_pred = model.predict(X)

print("Predictions:", y_pred)
print("Actual:", y)

r2 = model.score(X, y)
print(r2)
