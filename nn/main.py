from layers.dense import Dense
from models.model import Model
from layers.activations import ReLU
from losses.mse import MSELoss
from losses.binary_cross_entropy import BinaryCrossEntropy
import numpy as np
from optimizers.sgd import SGD

X = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

y = np.array([
    [2.0],
    [4.0],
    [6.0],
    [8.0],
    [10.0]
])


model = Model()
model.add(Dense(1,2))
model.add(ReLU())
model.add(Dense(2,1))
MSE = MSELoss()
BCE = BinaryCrossEntropy()

optimizer = SGD(model.layers)

for i in range(10):
    y_pred = model.forward(X)
    loss = BCE.forward(y_pred, y)
    print(y_pred)
    back = BCE.backward(y_pred, y)
    model.backward(back)
    optimizer.step()
