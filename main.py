# main.py
import numpy as np
from linear_model.logistic_regression import LogisticRegression

X = [[0.5, 1.2],
     [1.0, 2.0],
     [1.5, 2.8],
     [3.0, 4.0]]

y = [0, 1, 0, 2]

model = LogisticRegression(epoch=100, optimizer="batch")
model.fit(X, y)
acc =model.score(X, y)
print("Predictions:", model.predict(X))
print("Probabilities:", model.predict_proba(X))
print(acc)
