import numpy as np
class Dense:
    def __init__(self, input_units, output_units):
        self.input_units = input_units
        self.output_units = output_units
        self.W = np.random.randn(input_units, output_units)
        self.b = np.zeros(output_units)

    def forward(self, X):
        self.X = X
        Z = X @ self.W + self.b
        return Z

    def backward(self, dZ):
        self.dW = self.X.T @ dZ / self.X.shape[0]
        self.db = np.mean(dZ, axis=0)
        dX = dZ @ self.W.T
        return dX
