# main.py
import numpy as np
from linear_model.linear_regression import LinearRegression  # replace with your file name if different

# Sample data: 3 samples, 2 features
X = np.array([[1,2],[2,3],[3,4]])
y = np.array([3,5,7])


# Initialize and train model
model = LinearRegression(epoch=100)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Print results
print("Predictions:", y_pred)
print("Actual:", y)

r2 = model.score(X, y)
print("R² score:", r2)
