import numpy as np
class ReLU:
    def forward(self, X: np.ndarray):
        self.mask = X > 0
        masked_X = X * self.mask
        return masked_X
    
    def backward(self, dZ):
        output = self.mask * dZ
        return output
