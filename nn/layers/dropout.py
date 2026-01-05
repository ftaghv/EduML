import numpy as np
class Dropout:
    def __init__(self, p):
        self.p = p

    def forward(self, X):
        self.mask = (np.random.rand(*X.shape) < self.p).astype(int)
        output = X * self.mask / self.p
        return output

    def backward(self, dZ):
        output = dZ * self.mask / self.p
        return output
