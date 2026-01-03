class SGD:
    def __init__(self, layers):
        self.layers = layers
        self.lr = 0.1

    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'W'):
                layer.W = layer.W - self.lr * layer.dW
                layer.b = layer.b - self.lr * layer.db
