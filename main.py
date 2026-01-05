import numpy as np
from linear_model.linear_regression import Ridge, Lasso, ElasticNet, LinearRegression

X = np.array([[0.5],
     [1.0],
     [1.5],
     [3.0]])

y  = np.array([0.3, 1.2, 0.2, 2])

model = LinearRegression(100)
model.fit(X,y)
print(model.coef_)
#print(model.intercept_)
model = Ridge(100, 10)
model.fit(X,y)
print(model.coef_)
#print(model.intercept_)
