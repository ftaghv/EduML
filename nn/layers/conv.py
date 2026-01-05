import numpy as np
class ConV2D:
    def __init__(self, kernel_size, stride=None, padding=None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = np.random.randn(kernel_size, kernel_size)

    def forward(self, X):
        self.X = X
        out_shape = self.X.shape[1] - self.kernel_size + 1
        out = np.zeros((out_shape, out_shape))
        for i in range(out_shape):
            for j in range(out_shape):
                mat = self.X[i:i+self.kernel_size, j:j+self.kernel_size]
                out[i, j] = np.sum(np.multiply(mat, self.W))
        return out
    
    def backward(self, dZ):
        out_dW = dZ.shape[1]
        self.dW = np.zeros((self.kernel_size, self.kernel_size))
        for i in range(out_dW):
            for j in range(out_dW):
                X_patch = self.X[i:i+self.kernel_size, j:j+self.kernel_size]
                self.dW += (dZ[i, j] * X_patch)

        row, col = self.X.shape
        self.dX = np.zeros((row, col))
        for i in range(row - self.kernel_size + 1):
            for j in range(col - self.kernel_size + 1):
                mat = self.X[i:i+self.kernel_size, j:j+self.kernel_size]
                self.dX[i:i+self.kernel_size, j:j+self.kernel_size] += dZ[i,j] * np.flip(self.W, axis=(0,1))
        return self.dX
                 
