import numpy as np

class LogisticRegression:
    def __init__(self, epoch, optimizer, multi_class="auto"):
        self.epoch = epoch
        self.optimizer = optimizer
        self.multi_class = multi_class
        self.lr = 0.01
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        if self.multi_class == "auto":
            n_classes = len(np.unique(y))
            if n_classes == 2:
                self.multi_class = "binary"
            else:
                self.multi_class = "multinomial"
        if self.multi_class == "binary":
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
        if self.multi_class == "multinomial":
            X = np.array(X)
            y = np.array(y)
            n_classes = len(np.unique(y))
            n_samples = X.shape[0]
            n_features = X.shape[1]
            self.coef_ = np.zeros((n_features, n_classes))
            self.intercept_ = np.zeros(n_classes)
            y = np.eye(n_classes)[y]
            if self.optimizer == "batch":
                for i in range(self.epoch):
                    y_pred =  X @ self.coef_ + self.intercept_
                    y_pred = self._softmax(y_pred)
                    error = y_pred - y
                    self.coef_ = self.coef_ - self.lr * ((X.T @ error) / n_samples)
                    self.intercept_ = self.intercept_ - self.lr * np.mean(error, axis=0)

    def predict(self, X):
        if self.multi_class == "binary":
            X = np.array(X)
            y_pred = X @ self.coef_ + self.intercept_
            y_pred = self._sigmoid(y_pred)
            y_pred = (y_pred > 0.5).astype(int)
            return y_pred
        if self.multi_class == "multinomial":
            X = np.array(X)
            y_pred = X @ self.coef_ + self.intercept_
            y_pred = self._softmax(y_pred)
            y_pred = np.argmax(y_pred, axis=1)
            return y_pred
    
    def predict_proba(self, X):
        if self.multi_class == "binary":
            X = np.array(X)
            y_pred = X @ self.coef_ + self.intercept_
            y_pred = self._sigmoid(y_pred)
            return y_pred
        if self.multi_class == "multinomial":
            X = np.array(X)
            y_pred = X @ self.coef_ + self.intercept_
            y_pred = self._softmax(y_pred)
            return y_pred

    def _sigmoid(self, y_pred):
        return 1 / (1 + np.exp(- y_pred))
    
    def _softmax(self, y_pred):
        return np.exp(y_pred) / (np.sum(np.exp(y_pred), axis=1, keepdims=True))
    
    def score(self, x, y):
        if self.multi_class == "binary":
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
        if self.multi_class == "multinomial":
            X = np.array(x)
            y = np.array(y)
            y_pred = X @ self.coef_ + self.intercept_
            y_pred = self._softmax(y_pred)
            y_pred = np.argmax(y_pred, axis=1)
            n_samples = X.shape[0]
            true_pred = 0
            for i in range(len(y)):
                if y[i] == y_pred[i]:
                    true_pred += 1
            return true_pred / n_samples
