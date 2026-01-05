from layers.dense import Dense
from models.model import Model
from layers.activations import ReLU
from losses.mse import MSELoss
from losses.binary_cross_entropy import BinaryCrossEntropy
from layers.dropout import Dropout
import numpy as np
from optimizers.sgd import SGD
from layers.conv import ConV2D

X = np.array([
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5]
])

print(X[3:6, 3:6])

y = np.array([
    [2.0],
    [4.0],
    [6.0],
    [8.0],
    [10.0]
])


model = Model()
model.add(Dense(5,2))
model.add(ReLU())
model.add(Dropout(0.2))
model.add(Dense(2,1))
MSE = MSELoss()
BCE = BinaryCrossEntropy()

optimizer = SGD(model.layers)

for i in range(100):
    y_pred = model.forward(X) # all forward passes happen here
    loss = MSE.forward(y_pred, y)
    print(loss)
    back = MSE.backward(y_pred, y)
    model.backward(back)
    optimizer.step()
