import numpy as np

class MSELoss:
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self, y_pred, y_true):
        batch_size = y_true.shape[0]
        grad = (2 * (y_pred - y_true)) / batch_size
        return grad
