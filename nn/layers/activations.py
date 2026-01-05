import numpy as np
class ReLU:
    def __init__(self):
        pass

    def forward(self, X: np.ndarray):
        self.mask = X > 0
        masked_X = X * self.mask
        return masked_X
    
    def backward(self, dZ):
        output = self.mask * dZ
        return output

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, X: np.ndarray):
        self.output = 1 / (1 + np.exp(-X))
        return self.output
    
    def backward(self, dZ):
        grad = self.output * (1 - self.output)
        output = grad * dZ
        return output
