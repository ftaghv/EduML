import numpy as np

class LinearRegression:
    def __init__(self, epoch: int):
        self.epoch = epoch
        self.coef_ = None
        self.intercept_ = None
        self.lr = 0.1
        self.mean = None
        self.X_std = None

    def fit(self, X:np.ndarray , y:np.ndarray):
        X = np.array(X)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean # mean per feature
        self.X_std = np.std(X, axis=0)
        X_scaled = X_centered / self.X_std
        y = np.array(y)
        num_samples = X.shape[0]
        num_features = X.shape[1]
        self.coef_ = np.zeros(shape=num_features)
        self.intercept_ = 0
        for i in range(self.epoch):
            y_pred = X_scaled @ self.coef_ + self.intercept_
            error = y_pred - y
            self.coef_ = self.coef_ - self.lr * ((X_scaled.T @ error) / num_samples)
            self.intercept_ = self.intercept_ - self.lr * np.mean(error)

    def predict(self, X: np.ndarray):
        X = np.array(X)
        X_centered = X - self.mean
        X_scaled = X_centered / self.X_std
        y_pred = X_scaled @ self.coef_ + self.intercept_
        return y_pred

    def score(self, X, y):
        X = np.array(X)
        X_centered = X - self.mean
        X_scaled = X_centered / self.X_std
        y = np.array(y)
        y_pred = X_scaled @ self.coef_ + self.intercept_
        r_squared = 1 - (np.mean((y - y_pred) ** 2) / 
                         (np.mean((y - np.mean(y)) ** 2))) 
        return r_squared
