import numpy as np


class Perceptron:
    def __init__(self, num_features: int, epochs: int = 10):
        self.num_features = num_features
        self.weights = np.zeros((num_features, 1), dtype=np.float)
        self.bias = np.zeros(1, dtype=np.float)
        self._n_epochs = epochs

    def forward(self, x):
        linear = np.dot(x, self.weights) + self.bias
        predictions = np.where(linear > 0., 1, 0)
        return predictions

    def backward(self, x, y):
        predictions = self.forward(x)
        errors = y - predictions
        return errors

    def fit(self, x, y):
        for e in range(self._n_epochs):
            for i in range(y.shape[0]):
                errors = self.backward(x[i].reshape(1, self.num_features),
                                       y[i]).reshape(-1)
                self.weights += (errors * x[i]).reshape(self.num_features, 1)
                self.bias += errors

    def predict(self, x):
        return self.forward(x).reshape(-1)

    def evaluate(self, y, preds):
        return np.sum(preds == y) / y.shape[0]
