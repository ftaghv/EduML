from layers.dense import Dense
class Sequential:
    def __init__(self):
        self.layer = []

    def add(self, layer):
        self.layer.append(layer)

    def compile(self, loss="None"):
        pass

    def fit(self, X, y):
        for layer in self.layer:
            X_input = Dense.forward(X_input)
        y_pred = X_input
