import numpy as np
from linear_model.linear_regression import Ridge, Lasso

X = np.array([[0.5, 1.2],
     [1.0, 2.0],
     [1.5, 2.8],
     [3.0, 4.0]])

y  = np.array([0.3, 1.2, 0.2, 2])

model = Ridge(100, 10)
model.fit(X,y)
print(model.coef_)
#print(model.intercept_)
model = Lasso(100, 10)
model.fit(X,y)
print(model.coef_)
#print(model.intercept_)
