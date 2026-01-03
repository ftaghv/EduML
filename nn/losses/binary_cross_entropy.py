import numpy as np
class BinaryCrossEntropy:
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        loss = - np.mean((y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        return loss

    def backward(self, y_pred, y_true):
        batch_size = y_true.shape[0]
        grad =  ((- (y_true / y_pred)) + ((1 - y_true) / (1 - y_pred)) )/ batch_size
        return grad
