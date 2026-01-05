class Model:
    def __init__(self, layers=None):
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dZ):
        grad = dZ
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
