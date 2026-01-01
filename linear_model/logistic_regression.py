import numpy as np

class LogisticRegression:
    def __init__(self, epoch, optimizer):
        self.epoch = epoch
        self.optimizer = optimizer
        self.lr = 0.01
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        
        if self.optimizer == "batch":
            for i in range(self.epoch):
                y_pred =  X @ self.coef_ + self.intercept_
                y_pred = self._sigmoid(y_pred)
                error = y_pred - y
                self.coef_ = self.coef_ - self.lr * ((X.T @ error) / n_samples)
                self.intercept_ = self.intercept_ - self.lr * np.mean(error)

        if self.optimizer == "SGD":
            for i in range(self.epoch):
                for sample in range(n_samples):
                    y_pred =  X[sample] @ self.coef_ + self.intercept_
                    y_pred = self._sigmoid(y_pred)
                    error = y_pred - y[sample]
                    self.coef_ = self.coef_ - self.lr * (X[sample] * error)
                    self.intercept_ = self.intercept_ - self.lr * error

    def predict(self, X):
        X = np.array(X)
        y_pred = X @ self.coef_ + self.intercept_
        y_pred = self._sigmoid(y_pred)
        y_pred = (y_pred > 0.5).astype(int)
        return y_pred
    
    def predict_proba(self, X):
        X = np.array(X)
        y_pred = X @ self.coef_ + self.intercept_
        y_pred = self._sigmoid(y_pred)
        return y_pred
    
    def _sigmoid(self, y_pred):
        return 1 / (1 + np.exp(- y_pred))
    
    def score(self, x, y):
        X = np.array(x)
        y = np.array(y)
        y_pred = X @ self.coef_ + self.intercept_
        y_pred = self._sigmoid(y_pred)
        y_pred = (y_pred > 0.5).astype(int)
        n_samples = X.shape[0]
        true_pred = 0
        for i in range(len(y)):
            if y[i] == y_pred[i]:
                true_pred += 1
        return true_pred / n_samples
